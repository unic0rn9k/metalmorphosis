use std::{
    sync::{
        atomic::{AtomicIsize, AtomicU16, Ordering},
        Arc, Mutex, Weak,
    },
    thread::{self, JoinHandle},
};

use crate::{error::Result, Graph};

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

pub struct Pool {
    graph: Weak<Graph>,
    mpi_id: usize, // What machine does this pool live on?
    last_unoccupied: Mutex<OccupancyNode>,
    worker_handles: Vec<Arc<Worker>>,
    thread_handles: Vec<JoinHandle<()>>,
}

impl Pool {
    pub fn new(graph: Weak<Graph>) -> Arc<Self> {
        // TODO: Get the actual value here...
        let mpi_id = 0;

        let threads = std::thread::available_parallelism().unwrap().into();
        let mut worker_handles = vec![];

        for thread_id in 0..threads {
            let worker = Arc::new(Worker::new(DeviceID { mpi_id, thread_id }));
            worker_handles.push(worker);
        }

        Arc::new(Self {
            graph,
            mpi_id,
            last_unoccupied: Mutex::new(OccupancyNode::new()),
            thread_handles: vec![],
            worker_handles,
        })
    }

    pub unsafe fn init(self: &Arc<Self>) {
        // Some unholyness is still left tho
        let mut_self = &mut *(Arc::as_ptr(self) as *mut Self);

        // This is just gonna keep counting endlessly. For no reason...
        let initialized_threads = Arc::new(AtomicU16::new(0));
        let threads = self.worker_handles.len();

        for thread_id in 0..threads {
            let tc = initialized_threads.clone();
            let pool = self.clone();
            let graph = self
                .graph
                .upgrade()
                .expect("Graph dropped before Pool was initialized");
            let worker = self.worker_handles[thread_id].clone();

            mut_self.thread_handles.push(thread::spawn(move || {
                lock(&pool.last_unoccupied).insert(worker.home.clone(), &pool);
                tc.fetch_add(1, Ordering::Relaxed);

                loop {
                    let mut task = worker.task.load(Ordering::Acquire);
                    while task == -1 {
                        thread::park();
                        task = worker.task.load(Ordering::Acquire);
                    }

                    if task == -2 {
                        return;
                    }

                    graph.compute(task as usize);

                    worker.task.store(-1, Ordering::SeqCst);

                    lock(&pool.last_unoccupied).insert(worker.home.clone(), &pool);
                }
            }));
        }
        while initialized_threads.load(Ordering::Acquire) < threads as u16 {}
    }

    pub fn kill(self: &Arc<Self>) {
        //       Also how do we know that pools on other mpi instances aren't running?
        use std::panic;

        while self
            .worker_handles
            .iter()
            .any(|w| w.task.load(Ordering::SeqCst) >= 0)
        {}

        for n in 0..self.worker_handles.len() {
            self.worker_handles[n].task.store(-2, Ordering::SeqCst);
            self.thread_handles[n].thread().unpark();
        }

        for t in &self.thread_handles {
            while !t.is_finished() {}
        }

        let mut_self = unsafe { &mut *(Arc::as_ptr(self) as *mut Self) };
        while !self.thread_handles.is_empty() {
            if let Err(e) = mut_self.thread_handles.pop().unwrap().join() {
                panic::resume_unwind(e);
            }
        }
        assert_eq!(
            Arc::strong_count(self),
            1,
            "WorkPool killed, but there still exists references to it!"
        )
    }

    // TODO: This should take an IntoIter of tasks,
    pub fn assign(self: &Arc<Self>, task: impl IntoIterator<Item = usize>) {
        // TODO: Brug compare exchange til at tjekke om tasken allerede bliver polled, i stedet for at tjekke i compute.
        let mut occupancy = lock(&self.last_unoccupied);
        for task in task {
            println!("Assigning task");
            let device = occupancy.pop(self);
            if let Some(device) = device {
                println!("Found device for task:{task}");
                if device.mpi_id != self.mpi_id {
                    todo!("Another MPI instance")
                }
                let worker = &self.worker_handles[device.thread_id];
                worker.task.store(task as isize, Ordering::Release);
                self.thread_handles[device.thread_id].thread().unpark();
            } else {
                panic!(
                    "This is so sad. Were all OUT OF DEVICES. Thought there are still {} live threads",
                    self.thread_handles.iter().map(|t| !t.is_finished() as u32).sum::<u32>()
                )
            }
        }
    }
}
