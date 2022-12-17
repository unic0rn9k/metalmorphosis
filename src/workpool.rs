use std::{
    ops::{Deref, DerefMut},
    pin::Pin,
    ptr::null_mut,
    sync::{
        atomic::{fence, AtomicIsize, AtomicU16, Ordering},
        Arc, Mutex, Weak,
    },
    thread::{self, JoinHandle},
};

use crate::{error::Result, Graph};

//mod atomic_occupancy;
//use atomic_occupancy::*;
mod locked_occupancy;
use locked_occupancy::*;

#[derive(Clone)]
pub struct DeviceID {
    mpi_id: usize,
    thread_id: usize,
}

impl DeviceID {
    pub fn unoccupied(&self) -> OccupancyNode {
        OccupancyNode::from(self)
    }
}

pub struct Worker {
    task: AtomicIsize,
    prev_unoccupied: OccupancyNode,
    home: DeviceID,
}

impl Worker {
    pub fn new(home: DeviceID) -> Self {
        Self {
            task: AtomicIsize::new(-1),
            prev_unoccupied: home.unoccupied(),
            home,
        }
    }
}

// Oh god. What is this?
pub struct MutPtr<T>(*mut T);
unsafe impl<T> Send for MutPtr<T> {}
unsafe impl<T> Sync for MutPtr<T> {}
impl<T> MutPtr<T> {
    pub fn from<U>(s: &mut U) -> Self {
        Self(s as *mut U as *mut T)
    }
    pub fn null() -> Self {
        Self(null_mut())
    }
    pub fn is_null(self) -> bool {
        self.0.is_null()
    }
    pub fn ptr(self) -> *mut T {
        self.0
    }
    pub fn transmute<U>(self) -> MutPtr<U> {
        unsafe { MutPtr::from(&mut *self.0) }
    }
}

impl<T> Copy for MutPtr<T> {}
impl<T> Clone for MutPtr<T> {
    fn clone(&self) -> Self {
        Self(self.0)
    }
}

impl<T> Deref for MutPtr<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.0 }
    }
}

impl<T> DerefMut for MutPtr<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.0 }
    }
}

// TODO: There really is no reason for pool to live in an arg (except maybe for sub-graphs)
pub struct Pool {
    graph: Weak<Graph>,
    mpi_id: usize, // What machine does this pool live on?
    last_unoccupied: Mutex<OccupancyNode>,
    thread_handles: Vec<JoinHandle<()>>,
    worker_handles: Vec<Arc<Worker>>,
}
unsafe impl Send for Pool {}
unsafe impl Sync for Pool {}

// Pool har ikke brug for mutable access til self efter den er blevet initialized.
// Atomic operations er ikke mut. Thread operations er heller ikke mut.
// Bare wrap dit lort i Arc, og så skulle den være gucci?
impl Pool {
    pub fn new(graph: Weak<Graph>) -> Arc<Self> {
        Arc::new(Self {
            graph,
            mpi_id: 0,
            last_unoccupied: Mutex::new(OccupancyNode::new()),
            thread_handles: vec![],
            worker_handles: vec![],
        })
    }

    pub unsafe fn init(self: &Arc<Self>) {
        let threads = std::thread::available_parallelism().unwrap().into();
        let mut_self = &mut *(Arc::as_ptr(self) as *mut Self);

        // This is just gonna keep counting endlessly. For no reason...
        let initialized_threads = Arc::new(AtomicU16::new(0));

        for thread_id in 0..threads {
            let worker = Arc::new(Worker::new(self.device_id(thread_id)));
            mut_self.worker_handles.push(worker.clone());

            let tc = initialized_threads.clone();
            let pool = self.clone();
            let graph = self
                .graph
                .upgrade()
                .expect("Graph dropped before Pool was initialized");

            mut_self.thread_handles.push(thread::spawn(move || loop {
                //println!("Worker {} says Hi", worker.home.thread_id);

                fence(Ordering::SeqCst);
                lock(&pool.last_unoccupied).insert(worker.home.clone(), &pool);
                tc.fetch_add(1, Ordering::Relaxed);

                let mut task = worker.task.load(Ordering::Acquire);
                while task == -1 {
                    thread::park();
                    task = worker.task.load(Ordering::Acquire);
                    //fence(Ordering::SeqCst);
                }
                //fence(Ordering::SeqCst);

                //println!("{} Awoken to task:{task}", worker.home.thread_id);
                if task == -2 {
                    //println!("{} Dead", worker.home.thread_id);
                    return;
                }

                //#[cfg(test)]
                //println!("Polling {task} on thread:{}", worker.home.thread_id);

                graph.compute(task as usize);

                worker.task.store(-1, Ordering::SeqCst);
            }));
        }
        while initialized_threads.load(Ordering::Acquire) < threads as u16 {}
    }

    pub fn kill(self: &Arc<Self>) {
        //       Also how do we know that pools on other mpi instances arent running?
        use std::panic;

        while self
            .worker_handles
            .iter()
            .any(|w| w.task.load(Ordering::SeqCst) >= 0)
        {}

        for n in 0..self.worker_handles.len() {
            //println!("Killing {n}");

            //fence(Ordering::SeqCst);
            self.worker_handles[n].task.store(-2, Ordering::SeqCst);
            self.thread_handles[n].thread().unpark();
        }

        let mut_self = unsafe { &mut *(Arc::as_ptr(&self) as *mut Self) };
        while !self.thread_handles.is_empty() {
            if let Err(e) = mut_self.thread_handles.pop().unwrap().join() {
                panic::resume_unwind(e);
            }
        }
        assert_eq!(
            Arc::strong_count(&self),
            1,
            "WorkPool killed, but there still exists references to it!"
        )
    }

    pub fn is_this_device(&self, device: DeviceID) -> bool {
        self.mpi_id == device.mpi_id
    }

    // TODO: This should take an IntoIter of tasks,
    pub fn assign(self: &Arc<Self>, task: impl IntoIterator<Item = usize>) {
        // TODO: Brug compare exchange til at tjekke om tasken allerede bliver polled, i stedet for at tjekke i compute.
        let mut occupancy = lock(&self.last_unoccupied);
        for task in task {
            let device = occupancy.pop(self);
            if let Some(device) = device {
                if device.mpi_id != self.mpi_id {
                    todo!("Another MPI instance")
                }
                let worker = &self.worker_handles[device.thread_id];
                worker.task.store(task as isize, Ordering::Release);
                self.thread_handles[device.thread_id].thread().unpark();
            } else {
                panic!(
                "This is so sad. Were all OUT OF DEVICES. Thought there are still {} live threads",
                self.thread_handles.len()
            )
            }
        }
    }

    fn device_id(&self, thread_id: usize) -> DeviceID {
        DeviceID {
            mpi_id: self.mpi_id,
            thread_id,
        }
    }

    /*
    pub fn pop_device(self: &Arc<Self>) -> DeviceID {
        let bruh = match self.last_unoccupied.device() {
            Some(device) => {
                if device.mpi_id != self.mpi_id {
                    todo!("Another MPI instance")
                }
                self.last_unoccupied
                    .swap(&self.worker_handles[device.thread_id].last_unoccupied);
                device
            }
            None => {
                panic!("This is so sad. Were all OUT OF DEVICES. Thought there are still {} live threads", self.thread_handles.len())
            }
        };
        fence(Ordering::SeqCst);
        bruh
    }
    */
}
