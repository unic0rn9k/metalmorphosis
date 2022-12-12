use std::{
    ops::{Deref, DerefMut},
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
    last_available: AtomicIsize,
    thread_handles: Vec<JoinHandle<()>>,
    worker_handles: Vec<Worker>,
}
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
        unsafe { Self::init(PoolHandle::from(&mut tmp)) }
        tmp
    }

    unsafe fn init(mut pool: PoolHandle) {
        let threads = std::thread::available_parallelism().unwrap().into();
        for thread_id in 0..threads {
            let mut worker = Worker::new(DeviceID {
                mpi_id: 0,
                thread_id,
            });

            pool.worker_handles.push(worker);
            let worker = (*pool.ptr()).worker_handles.last_mut().unwrap();

            pool.clone().thread_handles.push(thread::spawn(move || {
                loop {
                    // TODO: Signal that this thread isn't doing anything.
                    let last_available = pool
                        .last_available
                        .swap(pool.last_available.load(Ordering::SeqCst), Ordering::SeqCst);
                    pool.last_available.store(last_available, Ordering::SeqCst);
                    let mut task = worker.task.load(Ordering::Acquire);
                    while task < 0 {
                        thread::park();
                        task = worker.task.load(Ordering::Acquire);
                    }

                    unsafe {
                        (*pool.graph).compute(task as usize, pool);
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
