#![feature(new_uninit)]
#![feature(get_mut_unchecked)]
#![feature(future_join)]

use rmp_serde as rmps;
use rmps::Serializer;
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::pin::Pin;
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};

mod stupid_futures;
use stupid_futures::*;
mod error;
use error::*;

pub type BasicFuture = Box<dyn Future<Output = ()> + Unpin>;
pub type OutputSlice = (*mut u8, usize);

pub struct TaskNode<T: Program> {
    sender: SyncSender<(T, usize, OutputSlice)>,
    output: OutputSlice,
    future: BasicFuture,
    parent: usize,
    this_node: usize,
    token: T,
    children: usize,
}

struct NilWaker;

impl Wake for NilWaker {
    fn wake(self: Arc<Self>) {
        panic!("Wake me up inside!");
    }
}

impl<T: Program> TaskNode<T> {
    pub fn write_output<O: MorphicIO>(&self, o: O) -> Result<(), T> {
        let buffer = unsafe { std::slice::from_raw_parts_mut(self.output.0, self.output.1) };
        Ok(o.serialize(&mut Serializer::new(buffer))?)
    }

    pub async fn branch<O: MorphicIO>(&self, token: T) -> Result<O, T> {
        println!("      +__");
        println!("      |  [{:?}]", token);
        println!("      |  ");
        let o_size = std::mem::size_of::<O>();
        let mut buffer = vec![0u8; o_size];
        self.sender
            .send((token, self.this_node, (&mut buffer[0] as *mut u8, o_size)))?;
        halt_once().await;
        Ok(rmps::from_slice(unsafe {
            std::mem::transmute(&mut buffer[..])
        })?)
    }

    pub fn poll(&mut self) -> Poll<()> {
        Pin::new(&mut self.future).poll(&mut Context::from_waker(&Waker::from(Arc::new(NilWaker))))
    }
}

pub trait MorphicIO: Serialize + Deserialize<'static> {
    //const SIZE: usize;
    //fn local_serialize(self, buffer: &mut [u8]);
    //fn local_deserialize(self, buffer: &[u8]);
}

pub trait Program: Copy + std::fmt::Debug {
    type Future: Future<Output = ()> + Unpin + 'static;
    fn future(&self, task_handle: Arc<TaskNode<Self>>) -> Self::Future;
    fn main() -> Self;
}

pub struct Executor<T: Program> {
    queue: Receiver<(T, usize, OutputSlice)>,
    self_sender: SyncSender<(T, usize, OutputSlice)>,
    task_graph: Vec<Arc<TaskNode<T>>>,
}

impl<T: Program> Executor<T> {
    pub fn new() -> Self {
        let (sender, reciever) = sync_channel(1000);
        let mut tmp = Self {
            queue: reciever,
            self_sender: sender,
            task_graph: vec![],
        };
        tmp.branch(T::main(), 0, ((&mut [] as *mut u8), 0));
        tmp
    }

    pub fn run(&mut self) -> Result<(), T> {
        let mut n = 0;

        'polling: loop {
            // Poll
            if n == self.task_graph.len() {
                n = 0;
                println!("Ran out of nodes. Fetching new ones...");
                let mut branch = self.queue.try_recv();
                while branch.is_ok() {
                    let (token, parent, output) = branch?;
                    self.branch(token, parent, output);
                    branch = self.queue.try_recv();
                }
            }

            if self.task_graph[n].children != 0 {
                n += 1;
                continue 'polling;
            }

            println!(
                "Polling node {:?} of {}",
                self.task_graph[n].token,
                self.task_graph.len()
            );
            if unsafe { Arc::get_mut_unchecked(&mut self.task_graph[n]) }
                .poll()
                .is_ready()
            {
                println!("Finished node {:?}", self.task_graph[n].token);
                if self.task_graph[n].this_node == self.task_graph[n].parent {
                    println!(
                        "DONE! {} nodes left behind: {:?}",
                        self.task_graph.len(),
                        self.task_graph.iter().map(|n| n.token).collect::<Vec<_>>()
                    );
                    self.task_graph.clear();
                    return Ok(());
                } else {
                    let parent = self.task_graph[n].parent;
                    self.task_graph.remove(n);
                    n = parent;
                    unsafe { Arc::get_mut_unchecked(&mut self.task_graph[n]).children -= 1 };
                    continue 'polling;
                }
            }

            println!("  ... [{:?}]\n", self.task_graph[n].token);

            n += 1;
        }
    }

    pub fn branch(&mut self, token: T, parent: usize, output: OutputSlice) {
        match self.task_graph.get_mut(parent) {
            None => {}
            Some(parent) => unsafe { Arc::get_mut_unchecked(parent).children += 1 },
        }
        let node = TaskNode {
            sender: self.self_sender.clone(),
            output,
            future: Box::new(UninitFuture),
            parent,
            this_node: self.task_graph.len(),
            token,
            children: 0,
        };
        self.task_graph.push(Arc::new(node));
        let last = self.task_graph.len() - 1;
        let node = &mut self.task_graph[last];
        let tmp = Box::new(token.future(Arc::clone(node)));
        unsafe { Arc::get_mut_unchecked(node) }.future = tmp;
    }
}

#[test]
fn basic() {
    use serde_derive::{Deserialize, Serialize};
    use std::future::{join, Future};

    impl MorphicIO for u32 {}

    #[derive(Serialize, Deserialize, Debug, Clone, Copy)]
    enum TestProgram {
        Main,
        A,
        B,
    }

    impl Program for TestProgram {
        type Future = Pin<Box<dyn Future<Output = ()>>>;

        fn future(&self, task_handle: Arc<TaskNode<Self>>) -> Self::Future {
            use TestProgram::*;
            match self {
                Main => Box::pin(async move {
                    println!("::start");
                    let a = task_handle.branch::<u32>(A);
                    let b = task_handle.branch::<u32>(B);
                    let (a, b) = join!(a, b).await;
                    println!("== {}", a.unwrap() + b.unwrap());
                    println!("::end");
                }),
                A => Box::pin(async move {
                    println!("::A");
                    task_handle.write_output(1).unwrap();
                }),
                B => Box::pin(async move {
                    println!("::B");
                    task_handle.write_output(2).unwrap();
                }),
            }
        }

        fn main() -> Self {
            TestProgram::Main
        }
    }

    let mut executor = Executor::<TestProgram>::new();
    executor.run().unwrap();
}
