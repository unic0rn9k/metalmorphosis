use std::{
    panic,
    sync::atomic::{AtomicIsize, Ordering},
    thread::{self, JoinHandle},
};

use crate::{error::Result, Graph};

const THREADS: usize = 4;

pub struct DeviceID {
    mpi_id: usize,
    thread_id: usize,
}

pub struct Worker {
    task: AtomicIsize,
    available_neighbor: AtomicIsize,
}

impl Worker {
    pub fn new() -> Self {
        Self {
            task: AtomicIsize::new(-1),
            available_neighbor: AtomicIsize::new(-1),
        }
    }
}
unsafe impl Send for Worker {}

pub struct MutPtr<T>(*mut T);
unsafe impl<T> Send for MutPtr<T> {}
impl<T> MutPtr<T> {
    unsafe fn get<'a>(&self) -> &'a mut T {
        &mut *self.0
    }
    pub fn from(s: &mut T) -> Self {
        Self(s as *mut T)
    }
    fn clone(&self) -> Self {
        Self(self.0)
    }
}
pub struct Pool {
    graph: MutPtr<Graph>,
    mpi_id: usize, // What machine does this pool live on?
    last_available: isize,
    thread_handles: Vec<JoinHandle<()>>,
    worker_handles: Vec<*mut AtomicIsize>,
}

impl Pool {
    pub fn new(graph: MutPtr<Graph>) -> Self {
        let mut tmp = Self {
            graph,
            mpi_id: 0,
            last_available: -1,
            thread_handles: vec![],
            worker_handles: vec![],
        };
        tmp.init();
        tmp
    }

    fn init(&mut self) {
        for n in 0..THREADS {
            let mut worker = Worker::new();
            let pool = self.handle();
            let graph = self.graph.clone();

            self.worker_handles.push(&mut worker.task as *mut _);

            self.thread_handles.push(thread::spawn(move || {
                let worker = worker;
                let mut task = worker.task.load(Ordering::Acquire);
                while task < 0 {
                    thread::park();
                    task = worker.task.load(Ordering::Acquire);
                }

                unsafe {
                    graph.get().compute(task as usize, pool);
                }
            }));
        }
    }

    pub fn kill(self) {
        //for worker in self.thread_handles {
        //    if let Err(e) = worker.join() {
        //        panic::resume_unwind(e);
        //    }
        //}
    }

    pub fn is_this_device(&self, device: DeviceID) -> bool {
        self.mpi_id == device.mpi_id
    }

    pub fn handle(&mut self) -> MutPtr<Self> {
        MutPtr(self as *mut _)
    }
}
