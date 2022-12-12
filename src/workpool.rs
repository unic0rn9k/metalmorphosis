use std::{
    ptr::null_mut,
    sync::atomic::{AtomicIsize, Ordering},
    thread::{self, JoinHandle},
};

use crate::{error::Result, Graph};

pub struct DeviceID {
    mpi_id: usize,
    thread_id: usize,
}

pub struct Worker {
    task: AtomicIsize,
    available_neighbor: DeviceID,
    home: DeviceID,
}

impl Worker {
    pub fn new(home: DeviceID) -> Self {
        Self {
            task: AtomicIsize::new(-1),
            available_neighbor: DeviceID {
                mpi_id: 0,
                thread_id: 0,
            },
            home,
        }
    }
}
unsafe impl Sync for Worker {}

#[derive(Copy, Clone)]
pub struct MutPtr<T>(*mut T);
unsafe impl<T> Sync for MutPtr<T> {}
unsafe impl<T> Send for MutPtr<T> {}
impl<T> MutPtr<T> {
    pub fn get(self) -> &'static mut T {
        unsafe { &mut *self.0 }
    }
    pub fn from<U>(s: &mut U) -> Self {
        Self(s as *mut U as *mut T)
    }
    fn clone(&self) -> Self {
        Self(self.0)
    }
    pub fn null() -> Self {
        Self(null_mut())
    }
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }
    pub fn ptr(self) -> *mut T {
        self.0
    }
    pub fn unholy<U>(self) -> MutPtr<U> {
        unsafe { MutPtr::from(&mut *self.0) }
    }
}

pub struct Pool {
    graph: *mut Graph,
    mpi_id: usize, // What machine does this pool live on?
    last_available: AtomicIsize,
    thread_handles: Vec<JoinHandle<()>>,
    worker_handles: Vec<Worker>,
}
unsafe impl Sync for Pool {}

pub type PoolHandle = MutPtr<Pool>;

impl Pool {
    pub fn new(graph: &mut Graph) -> Self {
        let mut tmp = Self {
            graph: graph as *mut _,
            mpi_id: 0,
            last_available: AtomicIsize::new(-1),
            thread_handles: vec![],
            worker_handles: vec![],
        };
        Self::init(PoolHandle::from(&mut tmp));
        tmp
    }

    fn init(pool: PoolHandle) {
        let threads = std::thread::available_parallelism().unwrap().into();
        for thread_id in 0..threads {
            let pool = pool.clone();
            let mut worker = Worker::new(DeviceID {
                mpi_id: 0,
                thread_id,
            });

            pool.clone().get().worker_handles.push(worker);
            let worker = pool.clone().get().worker_handles.last_mut().unwrap();

            pool.clone()
                .get()
                .thread_handles
                .push(thread::spawn(move || {
                    loop {
                        // TODO: Signal that this thread isn't doing anything.
                        let last_available = pool.clone().get().last_available.swap(
                            pool.clone().get().last_available.load(Ordering::SeqCst),
                            Ordering::SeqCst,
                        );
                        pool.clone()
                            .get()
                            .last_available
                            .store(last_available, Ordering::SeqCst);
                        let mut task = worker.task.load(Ordering::Acquire);
                        while task < 0 {
                            thread::park();
                            task = worker.task.load(Ordering::Acquire);
                        }

                        unsafe {
                            (*pool.clone().get().graph).compute(task as usize, pool.clone());
                        }
                    }
                }));
        }
    }

    pub fn kill(self) {
        // use std::panic;
        //for worker in self.thread_handles {
        //    if let Err(e) = worker.join() {
        //        panic::resume_unwind(e);
        //    }
        //}
    }

    pub fn is_this_device(&self, device: DeviceID) -> bool {
        self.mpi_id == device.mpi_id
    }

    pub fn handle(&mut self) -> PoolHandle {
        PoolHandle::from(self)
    }
}
