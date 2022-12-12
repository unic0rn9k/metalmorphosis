use std::{
    sync::atomic::AtomicIsize,
    thread::{self, JoinHandle},
};

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

pub struct Pool {
    mpi_id: usize, // What machine does this pool live on?
    last_available: isize,
    join_handles: Vec<JoinHandle<()>>,
    worker_handles: Vec<*mut AtomicIsize>,
}

impl Pool {
    pub fn new() -> Self {
        let mut tmp = Self {
            mpi_id: 0,
            last_available: -1,
            join_handles: vec![],
            worker_handles: vec![],
        };

        for n in 0..THREADS {
            let mut worker = Worker::new();

            tmp.worker_handles.push(&mut worker.task as *mut _);

            tmp.join_handles.push(thread::spawn(|| {
                worker;
                todo!();
            }));
        }

        tmp
    }

    pub fn is_this_device(&self, device: DeviceID) -> bool {
        self.mpi_id == device.mpi_id
    }
}
