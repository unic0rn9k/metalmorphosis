//! <div align="center">
//! <img src="https://raw.githubusercontent.com/unic0rn9k/metalmorphosis/4th_refactor/logo.png" width="400"/>
//! </div>
//!
//! Benchmarks can be found at [benchmarks_og_bilag.md](benchmarks_og_bilag.md)
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
    marker::PhantomData,
    pin::Pin,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{Receiver, Sender},
        Arc, RwLock,
    },
    task::{Context, Poll, Wake},
};

use buffer::Buffer;
use serde::{Deserialize, Serialize};
use workpool::Pool;

pub mod buffer;
//mod easy_api2;
pub mod dummy_net;
mod easy_api2;
pub mod error;
pub mod mpmc;
pub mod net;
pub mod workpool;

pub type BoxFuture = Pin<Box<dyn Future<Output = ()> + Send>>;

pub const DEBUG: bool = true;

/// # Safety
/// Graphs might be accessed mutably in parallel, even tho this is unsafe.
/// This will only be done for the IndexMut operation.
/// The executor will never access the same element, in parallel, assuming there are no duplicate `NodeId`s
pub unsafe trait Graph {
    fn len(&self) -> usize;

    fn task(&self, id: usize) -> &BoxFuture;
    fn task_mut(&mut self, id: usize) -> &mut BoxFuture;
}

pub struct Executor {
    graph: UnsafeCell<Box<dyn Graph>>,
    pool: Arc<Pool>,
}
unsafe impl Send for Executor {}
unsafe impl Sync for Executor {}

impl Executor {
    pub unsafe fn new(graph: impl Graph + 'static) -> Arc<Self> {
        Arc::new_cyclic(|exe| Executor {
            graph: UnsafeCell::new(Box::new(graph)),
            pool: Pool::new(exe.clone()),
        })
    }

    fn mpi_instance(&self) -> i32 {
        self.pool.mpi_instance()
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
                if reader.mpi_instance != self.mpi_instance() {
                    net.send(net::Event::Consumes {
                        awaited: node.clone(),
                        at: reader.mpi_instance,
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

    pub fn compute(self: &Arc<Self>, mut node: NodeId) {
        let waker = Arc::new(NilWaker).into();
        let mut cx = Context::from_waker(&waker);

        loop {
            if DEBUG {
                println!("Compputing {node:?}")
            };
            if node.0.done.load(Ordering::Acquire) {
                println!("already done");
                let continue_with = self.assign_children_of(&node);

                node.net()
                    .send(net::Event::NodeDone {
                        awaited: node.clone(),
                    })
                    .unwrap();

                node.0.is_being_polled.store(false, Ordering::Release);

                match continue_with {
                    Some(next) if next.try_poll() => node = next,
                    _ => return,
                }
                continue;
            }

            if node.mpi_instance != self.mpi_instance() {
                return;
            }

            match unsafe { Pin::new(&mut (*self.graph.get()).task_mut(node.this_node)) }
                .poll(&mut cx)
            {
                Poll::Ready(()) => {
                    println!("=== READY ===");
                    if node.this_node == 0 {
                        node.net().send(net::Event::Kill).unwrap()
                    }
                    node.0.done.store(true, Ordering::Release);
                }

                Poll::Pending => {
                    // TODO: Push to global que
                    if let Some(awaited) = unsafe { (*node.0.continue_to.get()).clone() } {
                        if awaited.mpi_instance != self.mpi_instance() {
                            node.net().send(net::Event::AwaitNode { awaited }).unwrap();
                            return;
                        }
                        node.0.is_being_polled.store(false, Ordering::Release);

                        if awaited.try_poll() {
                            return;
                        }

                        node = awaited
                    } else {
                        panic!("Pending nothing?");
                    }
                }
            }
        }
    }

    pub fn realize(self: &Arc<Self>, leafs: &[NodeId]) {
        self.pool.init(0);

        self.pool.assign(leafs.iter().filter_map(|n| {
            if n.mpi_instance == self.mpi_instance() {
                println!("leaf: {n:?}");
                Some(n.clone())
            } else {
                None
            }
        }));
    }

    pub fn kill(self: Arc<Self>, hold_on: Receiver<net::Event>) {
        self.pool.finish();
        drop(hold_on);
        match Arc::try_unwrap(self) {
            Ok(this) => this.pool.kill(),
            Err(s) => panic!(
                "Unable to gracefully kill graph execution, because there exists {} other references to it",
                Arc::strong_count(&s)
            ),
        }
    }
}

#[repr(align(128))]
pub struct Node {
    name: &'static str,
    continue_to: UnsafeCell<Option<NodeId>>,
    awaited_by: RwLock<mpmc::UndoStack<NodeId>>,
    this_node: usize,
    output: Buffer,
    done: AtomicBool,
    is_being_polled: AtomicBool,
    mpi_instance: i32,
    net_events: UnsafeCell<Option<Sender<net::Event>>>, // X
}
impl Node {
    pub fn new<T: Serialize + Deserialize<'static> + Sync + 'static>(this_node: usize) -> Node {
        Node {
            this_node,
            awaited_by: RwLock::new(mpmc::Stack::new(1, 3).undoable()),
            name: "NIL",
            continue_to: UnsafeCell::new(None),
            output: Buffer::new::<T>(),
            done: AtomicBool::new(false),
            is_being_polled: AtomicBool::new(false),
            mpi_instance: 0,
            net_events: UnsafeCell::new(None),
        }
    }
    pub fn commit(self) -> NodeId {
        NodeId(Arc::new(self))
    }

    pub unsafe fn use_net(&self, net: Option<Sender<net::Event>>) {
        unsafe { (*self.net_events.get()) = net }
    }

    fn net(&self) -> Sender<net::Event> {
        unsafe { (*self.net_events.get()).clone().expect("Network not set") }
    }

    // If this was not in Node, then check associated with downcasting would not be required.
    pub fn output<T: 'static>(&self) -> *mut T {
        unsafe { self.output.downcast_ptr_mut() }
    }

    fn try_poll(self: &Self) -> bool {
        self.is_being_polled.swap(true, Ordering::Acquire)
    }

    pub fn checkpoint(&self) {
        self.awaited_by.write().unwrap().checkpoint()
    }
    pub fn respawn(&self) {
        while !self.done.load(Ordering::Acquire) {}
        while self.try_poll() {}
        self.awaited_by.write().unwrap().undo();
        self.is_being_polled.store(false, Ordering::Release)
    }
}
unsafe impl Send for Node {}
unsafe impl Sync for Node {}

pub struct Symbol<T> {
    awaiter: NodeId,
    awaited: NodeId,
    marker: PhantomData<T>,
}

#[derive(Clone)]
pub struct NodeId(Arc<Node>);

impl NodeId {
    pub unsafe fn edge_from<T>(self, awaited: Self) -> Symbol<T> {
        Symbol {
            awaiter: self,
            awaited,
            marker: PhantomData,
        }
    }
}

impl std::fmt::Debug for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "#{}@{}-{}", self.this_node, self.mpi_instance, self.name)
    }
}

impl std::ops::Deref for NodeId {
    type Target = Node;

    fn deref(&self) -> &Self::Target {
        self.0.deref()
    }
}

struct NilWaker;
impl Wake for NilWaker {
    fn wake(self: Arc<Self>) {
        todo!("Wake me up inside!")
    }
}

pub struct Reader<T>(pub *const T);
unsafe impl<T> Send for Reader<T> {}

impl<T: 'static> Future for Symbol<T> {
    type Output = Reader<T>;

    fn poll(
        self: std::pin::Pin<&mut Self>,
        _: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        let Self {
            awaited, awaiter, ..
        } = &*self;

        if awaited.done.load(Ordering::Acquire) {
            unsafe { *awaiter.continue_to.get() = None }
            unsafe { Poll::Ready(Reader(awaited.output.downcast_ptr())) }
        } else {
            if unsafe { &*awaiter.continue_to.get() }
                .as_ref()
                .map(|node| node.this_node)
                == Some(self.awaited.this_node)
            {
                return Poll::Pending;
            }
            unsafe { *awaiter.continue_to.get() = Some(awaiter.clone()) }

            awaited.awaited_by.read().unwrap().push(awaiter.clone(), 1);
            Poll::Pending
        }
    }
}
