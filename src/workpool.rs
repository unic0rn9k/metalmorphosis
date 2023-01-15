use std::{
    cell::UnsafeCell,
    sync::{
        atomic::{AtomicIsize, AtomicU16, AtomicUsize, Ordering},
        Arc, Mutex, Weak,
    },
    thread::{self, JoinHandle},
};

use crate::{error::Result, net, Executor, Graph, Node, NodeId, DEBUG};

mod locked_occupancy;
use locked_occupancy::*;

#[derive(Clone)]
pub struct ThreadID(usize);

impl ThreadID {
    pub fn unoccupied(&self) -> WorkerStack {
        WorkerStack::from(self)
    }
}

pub struct Worker {
    task: NodeId,
    prev_unoccupied: WorkerStack,
    this_thread: ThreadID,
}

impl Worker {
    pub fn new(home: ThreadID) -> Self {
        Self {
            task: AtomicIsize::new(-3),
            prev_unoccupied: home.unoccupied(),
            this_thread: home,
        }
    }
}

pub struct Pool {
    mpi_instance: UnsafeCell<i32>, // What machine does this pool live on?
    last_unoccupied: Arc<Mutex<WorkerStack>>,
    worker_handles: Vec<Arc<Worker>>,
    thread_handles: Vec<JoinHandle<()>>,
    pub parked_threads: AtomicUsize,
}
unsafe impl Sync for Pool {}

pub struct PoolHandle {
    pool: Arc<Pool>,
    graph: Arc<Executor>,
}

impl Pool {
    pub fn new(graph: Weak<Executor>) -> Arc<Self> {
        Arc::new_cyclic(|pool| {
            //let threads = 4;
            let threads = std::thread::available_parallelism().unwrap().into();
            let mut worker_handles = vec![];
            let mut thread_handles = vec![];
            let last_unoccupied = Arc::new(Mutex::new(WorkerStack::new()));

            for thread_id in 0..threads {
                let worker = Arc::new(Worker::new(ThreadID(thread_id)));

                // TODO: If the last thread has been parked, then poll from global que.
                worker_handles.push(worker.clone());

                let pool = pool.clone();
                let graph = graph.clone();
                let last_unoccupied = last_unoccupied.clone();

                thread_handles.push(thread::spawn(move || {
                    thread::park();
                    let pool = pool
                        .upgrade()
                        .expect("Pool was dropped before thread was initialized");
                    lock(&last_unoccupied).insert(worker.this_thread.clone(), &pool);
                    let graph = graph
                        .upgrade()
                        .expect("Graph dropped before Pool was initialized");
                    worker.task.store(-1, Ordering::Release);

                    loop {
                        let mut task = worker.task.load(Ordering::Acquire);
                        while task == -1 {
                            thread::park();
                            task = worker.task.load(Ordering::Acquire);
                        }

                        if task == -2 {
                            return;
                        }

                        pool.parked_threads.fetch_sub(1, Ordering::SeqCst);
                        graph.compute(task as usize);
                        worker.task.store(-1, Ordering::SeqCst);

                        lock(&pool.last_unoccupied).insert(worker.this_thread.clone(), &pool);
                        pool.parked_threads.fetch_add(1, Ordering::SeqCst);
                    }
                }));
            }

            Self {
                mpi_instance: UnsafeCell::new(0),
                parked_threads: AtomicUsize::new(threads),
                last_unoccupied,
                thread_handles,
                worker_handles,
            }
        })
    }
}

impl PoolHandle {
    pub fn init(self: &Arc<Self>, mpi: i32) {
        unsafe { *self.mpi_instance.get() = mpi }
        for thread in &self.thread_handles {
            thread.thread().unpark();
        }
        while self
            .worker_handles
            .iter()
            .any(|w| w.task.load(Ordering::SeqCst) == -3)
        {}
    }

    pub fn finish(self: &Arc<Self>) {
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
    }

    pub fn kill(self: Arc<Self>) {
        //       Also how do we know that pools on other mpi instances aren't running?
        use std::panic;

        let mut this = match Arc::try_unwrap(self) {
            Ok(this) => this,
            Err(_) => panic!("tried to kill Pool while it was in use"),
        };
        while !this.thread_handles.is_empty() {
            if let Err(e) = this.thread_handles.pop().unwrap().join() {
                panic::resume_unwind(e);
            }
        }
    }

    pub fn mpi_instance(&self) -> i32 {
        unsafe { *self.mpi_instance.get() }
    }

    // Executor tries to reuse same threads. But doesn't try to use same threads for the same tasks
    pub fn assign<'a>(self: &Arc<Self>, task: impl IntoIterator<Item = &'a Arc<Node>>) {
        // TODO: Keep track of parked threads, and only take tasks, until all threads have been filled.
        let mut occupancy = lock(&self.last_unoccupied);
        for task in task {
            if task.is_being_polled.swap(true, Ordering::Acquire) {
                if DEBUG {
                    println!("  already being polled")
                };

                // TODO: Push to global que
                continue;
            }
            //if task.mpi_instance != self.mpi_instance() {
            //    task.net()
            //        .send(net::Event::AwaitNode(task.clone()))
            //        .unwrap();
            //    continue;
            //}

            let device = occupancy.pop(self);
            if let Some(device) = device {
                let worker = &self.worker_handles[device.0];
                worker
                    .task
                    .store(task.this_node as isize, Ordering::Release);
                self.thread_handles[device.0].thread().unpark();
            } else {
                panic!(
                    "This is so sad. Were all OUT OF DEVICES. Thought there are still {} live threads",
                    self.thread_handles.iter().map(|t| !t.is_finished() as u32).sum::<u32>()
                )
                // TODO: Push to global que
            }
        }
    }

    pub fn num_threads(self: &Arc<Self>) -> usize {
        self.thread_handles.len()
    }
}
