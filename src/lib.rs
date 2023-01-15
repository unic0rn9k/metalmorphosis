//! <div align="center">
//! <img src="https://raw.githubusercontent.com/unic0rn9k/metalmorphosis/4th_refactor/logo.png" width="400"/>
//! </div>
//!
//! Benchmarks can be found at [benchmarks_og_bilag.md](benchmarks_og_bilag.md)
//!
//! ## Definitions
//! - Symbol: a type used to refer to a node,
//!   that can be bound to another node, returning a future to the output of a node.
//!   (it lets you specify edges in the computation graph)
//!
//! - Dealocks will be caused by:
//! `graph.attach_edge(Self::edge(graph));`,
//! `graph.spawn(F(Self::edge(graph)));`.
//!
//! ## TODO
//! - [X] Type checking
//! - [X] Buffers
//! - [X] impl Future for Symbol
//!
//! - [X] handle for graph with type information about the node calling it.
//!
//! - [X] Executor / schedular
//!     - Wakers? (wake me up inside)
//! - [X] multithreaded
//!     - join future that works with array of symbols
//!
//! - [X] Distribute (OpenMPI?)
//!     - don't time awaits inside node
//!     - reusing output in node would confuse executor
//!
//! - [ ] clean code (remove duplicate work)
//! - [ ] nicer API (ATLEAST for custom schedular)
//! - [ ] return Result everywhere
//! - [ ] if a child is pushed to pool, but all threads are occupied, prefer to poll it from the thread of the parent
//!
//! - [X] priority que.
//!     - Let users set priority
//!     - increase priority of awaited children
//!     - internal events as tasks
//!
//! - [ ] Benchmarks and tests
//!     - TRAVLT? just make a synthetic benchmark... `thread::sleep(Duration::from_millis(10))`
//!
//! ## Extra
//! - Resources can be used for different executor instances, at the same time, using PoolHandle and NetHandle.
//! - Anchored nodes (so that 0 isnt special. Then executor makes sure anchored nodes are done before kill)
//! - Mby do some box magic in Graph::output, so that MutPtr is not needed.
//! - Allocator reusablility for dynamic graphs
//! - Const graphs (lib.rs/phf)
//! - Time-complexity hints
//! - Static types for futures (allocate them on bump, and let node provide funktion pointer for polling)
//! - Graph serialization (need runtime typechecking for graph hot-realoading)
//! - Optional stack trace (basically already implemented this)
//! - Check for cycles when building graph
//! - Multiple backends for providing tasks (eg: shared object files, cranelift, fancy jit / hot-reloading)
//! - specialized optimisations based on graph structure, when initilizing (fx: combine multiple nodes, that only have a signle parent, into one node)

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
use workpool::Pool;

pub mod buffer;
//mod easy_api2;
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

pub struct Executor {
    graph: *mut dyn Graph,
    pool: Arc<Pool>,
}
unsafe impl Send for Executor {}
unsafe impl Sync for Executor {}

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
        self.pool.assign(children.iter().cloned());
        continue_with.cloned()
    }

    pub fn compute(self: &Arc<Self>, node: &NodeId) {
        let waker = Arc::new(NilWaker).into();
        let mut cx = Context::from_waker(&waker);

        loop {
            if DEBUG {
                println!("{} compputing {}", node.mpi_instance(), node.name())
            };
            if node.0.done.load(Ordering::SeqCst) {
                println!("already done");
                let continue_with = self.assign_children_of(node);

                node.net()
                    .send(net::Event::NodeDone {
                        awaited: node.clone(),
                    })
                    .unwrap();

                node.0.is_being_polled.store(false, Ordering::Release);

                match &continue_with {
                    Some(next) if next.try_poll() => node = next,
                    _ => return,
                }
                continue;
            }

            if node.mpi_instance() != self.mpi_instance() {
                return;
            }

            match unsafe { Pin::new(&mut (*self.graph).task_mut(node.this_node())) }.poll(&mut cx) {
                Poll::Ready(()) => {
                    println!("=== READY ===");
                    if node.this_node() == 0 {
                        node.net().send(net::Event::Kill).unwrap()
                    }
                    node.0.done.store(true, Ordering::SeqCst);
                }

                Poll::Pending => {
                    // TODO: Push to global que
                    if let Some(awaited) = node.0.continue_to {
                        if awaited.mpi_instance() != self.mpi_instance() {
                            node.net().send(net::Event::AwaitNode { awaited }).unwrap();
                            return;
                        }
                        node.0.is_being_polled.store(false, Ordering::Release);

                        if awaited.try_poll() {
                            return;
                        }

                        node = &awaited
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
    pub fn new<T: Serialize + Deserialize<'static> + Sync>(len: usize) -> Node {
        todo!()
    }
    pub fn commit(self) -> NodeId {
        NodeId(Arc::new(self))
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

    fn try_poll(self: &Self) -> bool {
        !self.0.is_being_polled.swap(true, Ordering::Acquire)
    }

    pub fn mpi_instance(&self) -> i32 {
        self.0.mpi_instance
    }

    pub fn name(&self) -> &'static str {
        self.0.name
    }

    pub fn this_node(&self) -> usize {
        self.0.this_node
    }
}

struct NilWaker;
impl Wake for NilWaker {
    fn wake(self: Arc<Self>) {
        todo!("Wake me up inside!")
    }
}
