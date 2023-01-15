#![cfg(test)]
#![feature(test)]
#![feature(new_uninit)]

use std::{
    cell::UnsafeCell,
    future::Future,
    pin::Pin,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::Sender,
        Arc, RwLock,
    },
    task::{Context, Poll, Wake},
};

use buffer::Buffer;
use serde::{Deserialize, Serialize};
use workpool::{Pool, PoolHandle};

pub mod buffer;
mod easy_api2;
pub mod error;
pub mod mpmc;
pub mod net;
pub mod workpool;

pub type BoxFuture = Pin<Box<dyn Future<Output = ()> + Send>>;

pub const DEBUG: bool = true;

/// # Safety
/// Graphs might be accessed mutably in parallel, even tho this is unsafe.
/// This will only be done for the IndexMut operation. The executor will never access the same element at the same time.
pub unsafe trait Graph {
    fn len(&self) -> usize;

    fn task(&self, id: usize) -> &BoxFuture;
    fn task_mut(&mut self, id: usize) -> &mut BoxFuture;
}

pub struct NetHandle {
    net: Sender<net::Event>,
    graph: Arc<Executor>,
}

pub struct Executor {
    graph: *mut dyn Graph,
    net: NetHandle,
    pool: PoolHandle,
}

impl Executor {
    fn mpi_instance(&self) -> i32 {
        todo!()
    }

    fn children(self: &Arc<Self>, node: &NodeId) -> Vec<NodeId> {
        let net = node.net();
        node.0
            .awaited_by
            .write()
            .unwrap()
            .into_iter()
            .filter_map(move |reader| {
                //println!(
                //    "{} awaited by {} on {}",
                //    node.name,
                //    reader.name,
                //    self.mpi_instance()
                //);
                if reader.mpi_instance() != self.mpi_instance() {
                    net.send(net::Event::Consumes {
                        awaited: node.clone(),
                        at: reader.mpi_instance(),
                    })
                    .unwrap();
                    return None;
                }
                // TODO: Don't assign it if its already done.
                if reader.0.done.load(Ordering::Acquire) {
                    return None;
                }
                Some(reader)
            })
            .collect()
    }

    fn assign_children_of(self: &Arc<Self>, node: &NodeId) -> Option<NodeId> {
        let children = self.children(node);
        let continue_with = children.last();
        self.pool.assign(children.iter());
        continue_with.copied()
    }

    pub fn compute(self: &Arc<Self>, node: NodeId) {
        let waker = Arc::new(NilWaker).into();
        let mut cx = Context::from_waker(&waker);

        let mut node = &self.nodes[node];

        loop {
            if DEBUG {
                println!("{} compputing {}", node.mpi_instance, node.name)
            };
            if node.done.load(Ordering::SeqCst) {
                println!("already done");
                let continue_with = self.assign_children_of(node);

                node.net()
                    .send(net::Event::NodeDone {
                        awaited: node.this_node,
                    })
                    .unwrap();

                node.is_being_polled.store(false, Ordering::Release);

                match continue_with {
                    Some(next) if next.try_poll() => node = next,
                    _ => return,
                }
                continue;
            }

            if node.mpi_instance != self.mpi_instance() {
                return;
            }

            match unsafe { Pin::new(&mut *node.future.get()) }.poll(&mut cx) {
                Poll::Ready(()) => {
                    println!("=== READY ===");
                    if node.this_node == 0 {
                        node.net().send(net::Event::Kill).unwrap()
                    }
                    node.done.store(true, Ordering::SeqCst);
                }

                Poll::Pending => {
                    // TODO: Push to global que
                    let awaited = node.continue_to.load(Ordering::Acquire);
                    if awaited >= 0 {
                        let awaited = awaited as usize;
                        if self.nodes[awaited].mpi_instance != self.mpi_instance() {
                            node.net().send(net::Event::AwaitNode { awaited }).unwrap();
                            return;
                        }
                        node.is_being_polled.store(false, Ordering::Release);

                        if self.nodes[awaited]
                            .is_being_polled
                            .swap(true, Ordering::Acquire)
                        {
                            return;
                        }

                        node = &self.nodes[awaited]
                    } else {
                        panic!("Pending nothing?");
                    }
                }
            }
        }
    }
}

pub struct Symbol {
    awaiter: NodeId,
    awaited: NodeId,
}

#[repr(align(128))]
pub struct Node {
    executor: Arc<Executor>,
    name: &'static str,
    continue_to: Option<NodeId>,
    awaited_by: RwLock<mpmc::UndoStack<NodeId>>,
    this_node: usize,
    output: Buffer,
    done: AtomicBool,
    is_being_polled: AtomicBool,
    mpi_instance: i32,
    net_events: UnsafeCell<Option<Sender<net::Event>>>, // X
}
impl Node {
    fn new<T: Serialize + Deserialize<'static> + Sync>(len: usize) -> Node {
        todo!()
    }
}
unsafe impl Send for Node {}
unsafe impl Sync for Node {}

#[derive(Clone)]
pub struct NodeId(Arc<Node>);

impl NodeId {
    pub fn edge_from(&self, awaited: &Self) -> Symbol {
        Symbol {
            awaiter: self.clone(),
            awaited: awaited.clone(),
        }
    }
    fn new<T: Serialize + Deserialize<'static> + Sync + 'static>(this_node: usize) -> Self {
        NodeId(Arc::new(Node {
            this_node,
            awaited_by: RwLock::new(mpmc::Stack::new(1, 3).undoable()),
            name: "NIL",
            continue_to: None,
            output: Buffer::new::<T>(),
            done: AtomicBool::new(false),
            is_being_polled: AtomicBool::new(false),
            mpi_instance: 0,
            net_events: UnsafeCell::new(None),
        }))
    }

    fn use_net(&self, net: Option<Sender<net::Event>>) {
        unsafe { (*self.0.net_events.get()) = net }
    }

    fn net(&self) -> Sender<net::Event> {
        unsafe { (*self.0.net_events.get()).clone().expect("Network not set") }
    }

    // If this was not in Node, then check associated with downcasting would not be required.
    fn output<T: 'static>(&self) -> *mut T {
        unsafe { self.0.output.transmute_ptr_mut() }
    }

    fn try_poll(self: &Arc<Self>) -> bool {
        !self.0.is_being_polled.swap(true, Ordering::Acquire)
    }

    pub fn mpi_instance(&self) -> i32 {
        self.0.mpi_instance
    }
}

struct NilWaker;
impl Wake for NilWaker {
    fn wake(self: Arc<Self>) {
        todo!("Wake me up inside!")
    }
}
