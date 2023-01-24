use std::{
    cell::UnsafeCell,
    ops::DerefMut,
    sync::{
        atomic::{AtomicIsize, AtomicUsize, Ordering},
        Arc, RwLock, Weak,
    },
    thread::{self, JoinHandle},
};

use crate::{mpsc, Graph, Node, DEBUG};

#[derive(Clone, Debug)]
pub struct ThreadId(usize);

//impl ThreadId {
//    pub fn unoccupied(&self) -> WorkerStack {
//        WorkerStack::from(self)
//    }
//}

pub struct Worker {
    task: AtomicIsize,
    //prev_unoccupied: WorkerStack,
    home: ThreadId,
}

impl Worker {
    pub fn new(home: ThreadId) -> Self {
        Self {
            task: AtomicIsize::new(-3),
            //prev_unoccupied: home.unoccupied(),
            home,
        }
    }
}

pub struct Pool {
    mpi_instance: UnsafeCell<i32>, // What machine does this pool live on?
    last_unoccupied: Arc<RwLock<mpsc::Stack<ThreadId>>>,
    worker_handles: Vec<Arc<Worker>>,
    thread_handles: Vec<JoinHandle<()>>,
    pub parked_threads: AtomicUsize,
}
unsafe impl Sync for Pool {}

impl Pool {
    pub fn new(graph: Weak<Graph>) -> Arc<Self> {
        Arc::new_cyclic(|pool: &Weak<Pool>| {
            let threads = 4;
            //let threads = std::thread::available_parallelism().unwrap().into();
            let mut worker_handles = vec![];
            let mut thread_handles = vec![];
            let last_unoccupied = Arc::new(RwLock::new(mpsc::Stack::new(8, 1)));

            for thread_id in 0..threads {
                let worker = Arc::new(Worker::new(ThreadId(thread_id)));

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
                    last_unoccupied.read().unwrap().push(worker.home.clone(), 0);
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
                            println!(">>-(X _ X)->");
                            return;
                        }

                        if DEBUG {
                            println!("++ Thread {} unparked", worker.home.0);
                        }
                        pool.parked_threads.fetch_sub(1, Ordering::Release);
                        graph.compute(&graph.nodes[task as usize]);
                        worker.task.store(-1, Ordering::Relaxed);
                        last_unoccupied.read().unwrap().push(worker.home.clone(), 0);

                        pool.parked_threads.fetch_add(1, Ordering::Release);
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

    pub fn init(self: &Arc<Self>, mpi: i32) {
        unsafe { *self.mpi_instance.get() = mpi }
        // Some unholyness is still left tho
        for thread in &self.thread_handles {
            thread.thread().unpark();
        }
        while self
            .worker_handles
            .iter()
            .any(|w| w.task.load(Ordering::Acquire) == -3)
        {}
        //black_box(self.last_unoccupied.write().unwrap().into_iter());
    }

    pub fn finish(self: &Arc<Self>) {
        while self
            .worker_handles
            .iter()
            .any(|w| w.task.load(Ordering::Acquire) >= 0)
        {}

        for n in 0..self.worker_handles.len() {
            self.worker_handles[n].task.store(-2, Ordering::Release);
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

    pub fn assign<'a>(self: &Arc<Self>, tasks: impl IntoIterator<Item = &'a Arc<Node>>) {
        let mut occupancy = self.last_unoccupied.write().unwrap();
        let mut assigned = 0;
        let mut tasks = tasks.into_iter();
        for task in &mut tasks {
            while !task.try_poll("Pool::assign") {}

            let device = occupancy.pop();
            if let Some(device) = device {
                let worker = &self.worker_handles[device.0];
                worker
                    .task
                    .store(task.this_node as isize, Ordering::Release);
                self.thread_handles[device.0].thread().unpark();
                assigned += 1;
            } else {
                panic!(
                    "Only had space for {assigned} tasks. There are {} parked threads. Occupancy: {:?}",
                    self.parked_threads.load(Ordering::SeqCst),
                    occupancy.deref_mut()
                )
            }
        }
    }

    pub fn num_threads(self: &Arc<Self>) -> usize {
        self.thread_handles.len()
    }
}
