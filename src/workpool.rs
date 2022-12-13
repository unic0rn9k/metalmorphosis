use std::{
    hint::black_box,
    ops::{Deref, DerefMut},
    ptr::null_mut,
    sync::atomic::{fence, AtomicIsize, Ordering},
    thread::{self, JoinHandle},
};

//struct AtomicIsize(isize);
//impl AtomicIsize {
//    fn load(&self, _: Ordering) -> isize {
//        let mut a = black_box(self.0);
//        while a != self.0 {
//            a = black_box(self.0)
//        }
//        a
//    }
//    fn store(&mut self, val: isize, _: Ordering) {
//        while self.0 != val {
//            self.0 = black_box(val);
//        }
//    }
//    fn swap(&mut self, val: isize, o: Ordering) -> isize {
//        let a = self.load(o);
//        self.store(val, o);
//        a
//    }
//    fn new(val: isize) -> Self {
//        Self(val)
//    }
//}

use crate::{error::Result, Graph};

pub struct DeviceID {
    mpi_id: usize,
    thread_id: usize,
}

impl DeviceID {
    pub fn unoccupied(&self) -> DeviceOccupancyNode {
        DeviceOccupancyNode {
            last_thread: AtomicIsize::new(self.thread_id as isize),
            last_mpi: AtomicIsize::new(self.mpi_id as isize),
        }
    }
}

pub struct DeviceOccupancyNode {
    last_thread: AtomicIsize,
    last_mpi: AtomicIsize,
}

impl DeviceOccupancyNode {
    pub fn device(&self) -> Option<DeviceID> {
        let mpi = self.last_mpi.load(Ordering::SeqCst);
        let thread = self.last_thread.load(Ordering::SeqCst);
        if mpi < 0 || thread < 0 {
            None
        } else {
            Some(DeviceID {
                mpi_id: mpi as usize,
                thread_id: thread as usize,
            })
        }
    }

    pub fn swap(&mut self, other: &mut Self) {
        let ord = Ordering::SeqCst;
        let mut mpi = self.last_mpi.load(ord);
        let mut thread = self.last_thread.load(ord);
        mpi = other.last_mpi.swap(mpi, ord);
        thread = other.last_thread.swap(thread, ord);
        self.last_mpi.store(mpi, ord);
        self.last_thread.store(thread, ord);
    }

    pub fn thread_local_swap(&mut self, other: &mut Self) {
        let ord = Ordering::SeqCst;
        let mut thread = self.last_thread.load(ord);
        thread = other.last_thread.swap(thread, ord);
        self.last_thread.store(thread, ord);
    }

    pub fn new() -> Self {
        DeviceOccupancyNode {
            last_thread: AtomicIsize::new(-1),
            last_mpi: AtomicIsize::new(-1),
        }
    }
}

pub struct Worker {
    task: AtomicIsize,
    last_unoccupied: DeviceOccupancyNode,
    home: DeviceID,
}

impl Worker {
    pub fn new(home: DeviceID) -> Self {
        Self {
            task: AtomicIsize::new(-1),
            last_unoccupied: home.unoccupied(),
            home,
        }
    }
}

// Oh god. What is this?
pub struct MutPtr<T>(*mut T);
unsafe impl<T> Send for MutPtr<T> {}
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

pub struct Pool {
    graph: *mut Graph,
    mpi_id: usize, // What machine does this pool live on?
    last_unoccupied: DeviceOccupancyNode,
    thread_handles: Vec<JoinHandle<()>>,
    worker_handles: Vec<Worker>,
}
pub type PoolHandle = MutPtr<Pool>;

impl Pool {
    pub fn new(graph: &mut Graph) -> Self {
        let mut tmp = Self {
            graph: graph as *mut _,
            mpi_id: 0,
            last_unoccupied: DeviceOccupancyNode::new(),
            thread_handles: vec![],
            worker_handles: vec![],
        };
        unsafe { Self::init(PoolHandle::from(&mut tmp)) }
        tmp
    }

    unsafe fn init(mut pool: PoolHandle) {
        // TODO: MPI distribute distribute!
        let threads = std::thread::available_parallelism().unwrap().into();
        //let threads = 4;
        for thread_id in 0..threads {
            let worker = Worker::new(pool.device_id(thread_id));
            pool.worker_handles.push(worker);

            let worker = &mut (*pool.ptr()).worker_handles[thread_id];

            pool.clone()
                .thread_handles
                .push(thread::spawn(move || loop {
                    println!("Worker {} says Hi", worker.home.thread_id);
                    pool.last_unoccupied.swap(&mut worker.last_unoccupied);

                    let mut task = worker.task.load(Ordering::Acquire);
                    while task == -1 {
                        thread::park();
                        task = worker.task.load(Ordering::Acquire);
                        fence(Ordering::SeqCst);
                    }
                    //fence(Ordering::SeqCst);

                    println!("{} Awoken to task:{task}", worker.home.thread_id);
                    if task == -2 {
                        println!("{} Dead", worker.home.thread_id);
                        return;
                    }

                    #[cfg(test)]
                    println!("Polling {task} on thread:{}", worker.home.thread_id);

                    unsafe {
                        (*pool.graph).compute(task as usize, pool);
                    }

                    worker.task.store(-1, Ordering::SeqCst);
                }));
        }
    }

    pub fn kill(mut self) {
        use std::panic;

        //for n in 0..self.worker_handles.len() {
        //    self.assign(-2);
        //}

        for n in 0..self.worker_handles.len() {
            println!("Killing {n}");
            //self.worker_handles[n].task.store(-2, Ordering::SeqCst);
            //self.thread_handles[n].thread().unpark();
            //self.worker_handles[n].task.store(-2, Ordering::SeqCst);

            //fence(Ordering::SeqCst);
            //while self.worker_handles[n].task.load(Ordering::Acquire) != -2 {
            //self.worker_handles[n].task.store(-2, Ordering::SeqCst);
            fence(Ordering::SeqCst);
            self.worker_handles[n].task.store(-2, Ordering::SeqCst);
            assert_eq!(self.worker_handles[n].task.load(Ordering::SeqCst), -2); // Virker

            //println!("...");
            //}
            //fence(Ordering::SeqCst);

            assert_eq!(self.worker_handles[n].task.load(Ordering::SeqCst), -2);
            self.thread_handles[n].thread().unpark();
            assert_eq!(self.worker_handles[n].task.load(Ordering::SeqCst), -2);
        }

        for n in 0..self.worker_handles.len() {
            println!("Killing {n}");
            self.thread_handles[n].thread().unpark();
            if let Err(e) = self.thread_handles.swap_remove(n).join() {
                panic::resume_unwind(e);
            }
        }
    }

    pub fn is_this_device(&self, device: DeviceID) -> bool {
        self.mpi_id == device.mpi_id
    }

    pub fn handle(&mut self) -> PoolHandle {
        PoolHandle::from(self)
    }

    pub fn assign(&mut self, task: isize) {
        let device = self.pop_device();
        let worker = &mut self.worker_handles[device.thread_id];
        worker.task.store(task as isize, Ordering::Release);
        self.thread_handles[device.thread_id].thread().unpark();

        let mut bruh = worker.task.load(Ordering::SeqCst);
        while bruh >= 0 {
            bruh = worker.task.load(Ordering::SeqCst);
        }
    }

    fn device_id(&self, thread_id: usize) -> DeviceID {
        DeviceID {
            mpi_id: self.mpi_id,
            thread_id,
        }
    }

    pub fn pop_device(&mut self) -> DeviceID {
        match self.last_unoccupied.device() {
            Some(device) => {
                if device.mpi_id != self.mpi_id {
                    todo!("Another MPI instance")
                }
                self.last_unoccupied
                    .swap(&mut self.worker_handles[device.thread_id].last_unoccupied);
                device
            }
            None => panic!("This is so sad. Were all OUT OF DEVICES"),
        }
    }
}
