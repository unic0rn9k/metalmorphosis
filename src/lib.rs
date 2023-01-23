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
//! - [ ] references in nodes
//! - [ ] nicer API (ATLEAST for custom schedular)
//!
//! - [ ] clean code (remove duplicate work)
//! - [ ] return Result everywhere
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

// # Schedular
// If task returns pending, poll node in sources,
// otherwise poll node in readers.
//
// This can be in the future impl for symbols.
// Then the symbol needs to check if the node is already being polled from somwhere else.
// (this could be on a completely different machine)
//
// If you poll a symbol for a node, that is done, and has no pending readers left,
// re-initialise its task.
//
// It should be possible to send/recv 3 kinds of messages over mpi
// - a request for node's output
// - a node's output value
#![cfg(test)]
#![feature(test)]
#![feature(new_uninit)]

pub mod builder;
mod dummy_net;
mod error;
pub mod mpsc;
mod net;
mod workpool;

//use dummy_net as net_;
use net as net_;

use error::{Error, Result};
use mpsc::UndoStack;
use workpool::Pool;

pub use mpi::time;
use serde::{Deserialize, Serialize};
use std::any::{type_name, Any};
use std::cell::{RefCell, UnsafeCell};
use std::future::Future;
use std::hint::spin_loop;
use std::marker::{PhantomData, PhantomPinned};
use std::mem::transmute;
use std::ops::DerefMut;
use std::pin::Pin;
use std::rc::Rc;
use std::sync::atomic::Ordering;
use std::sync::atomic::{AtomicBool, AtomicIsize};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, RwLock};
use std::task::{Context, Poll, Wake};

const DEBUG: bool = false;

#[derive(Clone, Copy)]
pub struct Symbol<T>(usize, PhantomData<T>);
#[derive(Clone, Copy)]
pub struct LockedSymbol<T> {
    returner: usize,
    reader: usize,
    marker: PhantomData<T>,
}

// use From<LockedSymbol> instead
impl<T> LockedSymbol<T> {
    fn own(self, graph: &Arc<Graph>) -> OwnedSymbol<T> {
        OwnedSymbol {
            returner: graph.nodes[self.returner].clone(),
            reader: graph.nodes[self.reader].clone(),
            marker: PhantomData,
        }
    }
}

#[derive(Clone)]
pub struct OwnedSymbol<T> {
    returner: Arc<Node>,
    reader: Arc<Node>,
    marker: PhantomData<T>,
}
unsafe impl<T> Send for OwnedSymbol<T> {}

#[derive(Clone)]
pub struct SymbolGroup<T> {
    symbols: Vec<OwnedSymbol<T>>,
}

//impl<T: 'static> Future for SymbolGroup<T>{
//    type Output = Reader<T>;
//
//}

pub struct Reader<T>(pub *const T);
unsafe impl<T> Send for Reader<T> {}

impl<T: 'static> Future for OwnedSymbol<T> {
    type Output = Reader<T>;

    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        if DEBUG {
            println!(
                "polling: {:?} for {:?}",
                self.returner.this_node, self.reader.this_node,
            )
        };
        if self.returner.done.load(Ordering::Acquire) {
            if DEBUG {
                println!("task was done")
            };
            self.reader.continue_to.store(-1, Ordering::Release);
            unsafe { Poll::Ready(Reader((*self.returner.output.get()).ptr() as *const T)) }
        } else {
            if DEBUG {
                println!("task was pending {}", self.returner.this_node)
            };
            if self.reader.continue_to.load(Ordering::Acquire) == self.returner.this_node as isize {
                return Poll::Pending;
            }
            self.reader
                .continue_to
                .store(self.returner.this_node as isize, Ordering::Release);
            self.returner
                .awaited_by
                .read()
                .unwrap()
                .push(self.reader.this_node, 1);
            Poll::Pending
        }
    }
}

pub type BoxFuture = Pin<Box<dyn Future<Output = ()> + Send>>;
pub type AsyncFunction = Box<dyn Fn(&Arc<Graph>, Arc<Node>) -> BoxFuture>;

// Node could be split into two structs. One for everything that needs unsafe interior mutability. And one for the other stuff.
// TODO: Benchmark with and without repr
//#[repr(align(128))]
pub struct Node {
    name: &'static str,
    task: AsyncFunction,
    future: UnsafeCell<BoxFuture>, // X
    continue_to: AtomicIsize,      // This is just a Waker...
    awaited_by: RwLock<mpsc::UndoStack<usize>>,
    this_node: usize,
    output: UnsafeCell<Buffer>,
    done: AtomicBool,
    is_being_polled: AtomicBool,
    mpi_instance: i32,
    net_events: UnsafeCell<Option<Sender<net::Event>>>, // X
}
unsafe impl Send for Node {}
unsafe impl Sync for Node {}

impl Node {
    fn new<T: Serialize + Deserialize<'static> + Sync + 'static>(
        this_node: usize,
        out: Option<*mut T>,
    ) -> Self {
        Node {
            this_node,
            awaited_by: RwLock::new(mpsc::Stack::new(1, 3).undoable()),
            name: "NIL",
            task: Box::new(|_, _| Box::pin(async {})),
            future: UnsafeCell::new(Box::pin(async {})),
            continue_to: AtomicIsize::new(-1),
            output: UnsafeCell::new(Buffer::new::<T>(out)),
            done: AtomicBool::new(false),
            is_being_polled: AtomicBool::new(false),
            mpi_instance: 0,
            net_events: UnsafeCell::new(None),
        }
    }

    fn respawn(self: &Arc<Self>, graph: &Arc<Graph>) {
        unsafe {
            while !self.try_poll("Respawn") {}
            self.awaited_by.write().unwrap().undo();
            (*self.future.get()) = (self.task)(graph, self.clone());
            self.done.store(false, Ordering::Release);
            self.is_being_polled.store(false, Ordering::Release);
            assert!(!self.is_being_polled.load(Ordering::Acquire));
        }
    }

    fn use_net(self: &Arc<Self>, net: Option<Sender<net::Event>>) {
        unsafe { (*self.net_events.get()) = net }
    }

    fn net(self: &Arc<Self>) -> Sender<net::Event> {
        unsafe { (*self.net_events.get()).clone().expect("Network not set") }
    }

    // If this was not in Node, then check associated with downcasting would not be required.
    fn output<T: 'static>(self: &Arc<Self>) -> *mut T {
        unsafe { &mut *((*self.output.get()).mut_ptr() as *mut T) }
    }

    fn try_poll(self: &Arc<Self>, src: &str) -> bool {
        if DEBUG {
            println!(
                "trying to poll {}::{} from {}",
                self.this_node, self.name, src
            );
        }
        !self.is_being_polled.swap(true, Ordering::Acquire)
    }
}

pub struct Buffer {
    data: *mut dyn Any,
    de: fn(&[u8], *mut ()) -> Result<()>,
    se: fn(*const ()) -> Result<Vec<u8>>,
    drop: fn(&mut Self),
}

impl Drop for Buffer {
    fn drop(&mut self) {
        let d = self.drop;
        d(self);
    }
}

impl Buffer {
    pub fn new<T: Serialize + Deserialize<'static> + Sync + 'static>(src: Option<*mut T>) -> Self {
        unsafe {
            let data;
            let drop: fn(&mut Buffer);
            if let Some(src) = src {
                data = src;
                drop = |_| {};
            } else {
                data = Box::into_raw(Box::<T>::new_zeroed().assume_init());
                drop = |this: &mut Buffer| std::mem::drop(Box::from_raw(this.data));
            }
            Buffer {
                data,
                de: |b, out| {
                    *(out as *mut T) = bincode::deserialize(transmute(b))?;
                    Ok(())
                },
                se: |v| Ok(bincode::serialize::<T>(&*(v as *const T))?),
                drop,
            }
        }
    }

    pub fn serialize(&self) -> Vec<u8> {
        (self.se)(self.ptr()).expect("Buffer serialization failed")
    }

    pub fn deserialize(&mut self, data: &[u8]) {
        (self.de)(data, self.mut_ptr()).expect("Buffer deserialization failed")
    }

    fn ptr(&self) -> *const () {
        unsafe { transmute::<*mut dyn Any, (*const (), &())>(self.data).0 }
    }
    fn mut_ptr(&mut self) -> *mut () {
        unsafe { transmute::<*mut dyn Any, (*mut (), &())>(self.data).0 }
    }
}

pub struct Graph {
    // If this allocator is made reusable between graphs,
    // it would be safe to create a new graph inside an async block
    // and return a locked symbol from it. (also would require reusing the mpi universe)
    nodes: Vec<Arc<Node>>,
    _marker: PhantomPinned,
    pool: Arc<Pool>,
    leafs: Arc<RwLock<mpsc::UndoStack<usize>>>,
    // sub_graphs: Vec<GraphSpawner>
}
unsafe impl Sync for Graph {}
unsafe impl Send for Graph {}

impl Graph {
    pub fn from_nodes(
        nodes: Vec<Arc<Node>>,
        leafs: Arc<RwLock<mpsc::UndoStack<usize>>>,
    ) -> Arc<Self> {
        Arc::new_cyclic(|graph| Graph {
            nodes,
            _marker: PhantomPinned,
            pool: Pool::new(graph.clone()),
            leafs,
        })
    }

    //pub fn extend(self: &Arc<Self>, nodes: Vec<Arc<Node>>) -> Arc<Self> {
    //    Arc::new(Graph {
    //        nodes,
    //        _marker: PhantomPinned,
    //        pool: self.pool.clone(),
    //    })
    //}

    pub fn mpi_instance(self: &Arc<Self>) -> i32 {
        self.pool.mpi_instance()
    }

    fn children<'a>(self: &'a Arc<Self>, node: &'a Arc<Node>) -> Vec<&'a Arc<Node>> {
        let net = node.net();
        node.awaited_by
            .write()
            .unwrap()
            .into_iter()
            .filter_map(move |i| {
                let reader = &self.nodes[i];
                //if DEBUG {
                //}
                if reader.mpi_instance != self.mpi_instance() {
                    net.send(net::Event::Consumes {
                        awaited: node.this_node,
                        at: reader.mpi_instance,
                    })
                    .unwrap();
                    return None;
                }
                // TODO: Don't assign it if its already done.
                if reader.done.load(Ordering::Acquire) {
                    return None;
                }
                if DEBUG {
                    println!("{} polled by parent {}", reader.name, node.name,);
                }
                Some(reader)
            })
            .collect()
    }

    fn assign_children_of<'a>(self: &'a Arc<Self>, node: &'a Arc<Node>) -> Option<&'a Arc<Node>> {
        let mut children = self.children(node);
        let continue_with = children.pop();
        self.pool.assign(children.iter().copied());
        continue_with
    }

    pub fn compute<'a>(self: &'a Arc<Self>, mut node: &'a Arc<Node>) {
        let waker = Arc::new(NilWaker).into();
        let mut cx = Context::from_waker(&waker);

        loop {
            if DEBUG {
                println!("compputing {}::{}", node.this_node, node.name)
            };
            if node.done.load(Ordering::Acquire) {
                if DEBUG {
                    println!("already done")
                }
                let continue_with = self.assign_children_of(node);

                node.net()
                    .send(net::Event::NodeDone {
                        awaited: node.this_node,
                    })
                    .unwrap();

                node.is_being_polled.store(false, Ordering::Release);

                match continue_with {
                    Some(next) if next.try_poll("Continue to child") => node = next,
                    _ => return,
                }
                continue;
            }

            if node.mpi_instance != self.mpi_instance() {
                return;
            }

            match unsafe { Pin::new(&mut *node.future.get()) }.poll(&mut cx) {
                Poll::Ready(()) => {
                    if DEBUG {
                        println!("* {} READY", node.this_node)
                    }
                    if node.this_node == 0 {
                        node.net().send(net::Event::Kill).unwrap()
                    }
                    node.done.store(true, Ordering::Release);
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

                        while !self.nodes[awaited].try_poll("Continue to parent") {
                            // FIXME: this should not be blocking
                            //node.awaited_by.read().unwrap().push(awaited, 0);
                            //return;
                        }

                        node = &self.nodes[awaited]
                    } else {
                        panic!("Pending nothing?");
                    }
                }
            }
        }
    }

    pub fn init_net(self: &Arc<Self>) -> (Sender<net::Event>, net_::Networker) {
        assert_eq!(
            Arc::strong_count(self),
            1,
            "Cannot init Graph, because there exists {} other references to it",
            Arc::strong_count(self),
        );
        let (s, n) = net_::instantiate(self.clone());
        self.pool.init(n.rank());
        (s, n)
    }

    pub fn realize(self: &Arc<Self>, net_events: Sender<net::Event>) {
        self.leafs.write().unwrap().undo();
        for n in &self.nodes {
            n.respawn(self);
            n.use_net(Some(net_events.clone()));
        }

        if DEBUG {
            println!(
                "Leafs: {:?}",
                self.leafs.write().unwrap().deref_mut().deref_mut()
            );
        }

        let mut leaves = self.leafs.write().unwrap();
        let mut leaves = leaves.into_iter();
        let continue_with = leaves.next();
        self.pool.assign(leaves.filter_map(|n| {
            if DEBUG {
                println!("LEAF: {n}")
            }
            if self.nodes[n].mpi_instance == self.mpi_instance() {
                Some(&self.nodes[n])
            } else {
                None
            }
        }));
        if let Some(node) = continue_with {
            let node = &self.nodes[node];
            if node.try_poll("Root") {
                self.compute(node)
            }
        }
        //while !self.nodes[0].done.load(Ordering::Acquire) {
        //    while !self.nodes[0].try_poll("Main spin") {}
        //    self.compute(&self.nodes[0])
        //}
        //if self.mpi_instance() == 0 {
        //    self.pool.assign([self.nodes[0].clone()])
        //}
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

    pub fn spin_down(self: &Arc<Self>) {
        while self.pool.parked_threads.load(Ordering::Acquire) != self.pool.num_threads() {
            spin_loop()
        }
    }

    //pub fn print(&self) {
    //    for node in &self.nodes {
    //        //if node.is_being_polled.swap(true, Ordering::Acquire) {
    //        //    if DEBUG{println!("{} -> {} is being polled", node.mpi_instance, node.name)};
    //        //    return;
    //        //}
    //        let mut awaiters = vec![];
    //        for n in node.awaited_by.try_iter() {
    //            awaiters.push((n, self.nodes[n].name));
    //        }
    //        for (n, _) in &awaiters {
    //            node.awaiter.send(*n).unwrap();
    //        }
    //        if DEBUG {
    //            println!(
    //                "{} -> {} awaited by {:?}",
    //                node.mpi_instance, node.name, awaiters
    //            )
    //        };
    //        node.is_being_polled.store(false, Ordering::Release);
    //    }
    //}
}

struct NilWaker;
impl Wake for NilWaker {
    fn wake(self: Arc<Self>) {
        todo!("Wake me up inside!")
    }
}

#[cfg(test)]
mod blur;
