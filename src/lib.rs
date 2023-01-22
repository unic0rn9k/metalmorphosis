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

mod dummy_net;
mod error;
pub mod mpmc;
mod net;
mod workpool;

use dummy_net as net_;
//use net as net_;

use error::{Error, Result};
use mpmc::UndoStack;
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
use std::sync::mpsc::{channel, Receiver, Sender};
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
pub struct LockedSymbolGroup<T> {
    returners: Vec<usize>,
    reader: usize,
    marker: PhantomData<T>,
}

impl<T> LockedSymbolGroup<T> {
    fn own(self, graph: &Arc<Graph>) -> OwnedSymbolGroup<T> {
        OwnedSymbolGroup {
            returners: self
                .returners
                .iter()
                .map(|r| graph.nodes[*r].clone())
                .collect(),
            reader: graph.nodes[self.reader].clone(),
            marker: PhantomData,
        }
    }
}

#[derive(Clone)]
pub struct OwnedSymbolGroup<T> {
    returners: Vec<Arc<Node>>,
    reader: Arc<Node>,
    marker: PhantomData<T>,
}
unsafe impl<T> Send for OwnedSymbolGroup<T> {}

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

/// The first argument is a vector containing all the parents of the node being scheduled.
/// The second argument is the amount of mpi instances.
/// The output should be the mpi instance that the task will be executed on.
/// The output must never be greater than the amount of mpi instances.
pub type Scheduler = fn(Vec<&Node>, usize) -> usize;

pub fn keep_local_schedular(_: Vec<&Node>, _: usize) -> usize {
    0
}

// Node could be split into two structs. One for everything that needs unsafe interior mutability. And one for the other stuff.
// TODO: Benchmark with and without repr
#[repr(align(128))]
pub struct Node {
    name: &'static str,
    task: AsyncFunction,
    future: UnsafeCell<BoxFuture>, // X
    continue_to: AtomicIsize,      // This is just a Waker...
    awaited_by: RwLock<mpmc::UndoStack<usize>>,
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
            awaited_by: RwLock::new(mpmc::Stack::new(1, 3).undoable()),
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
            while self.is_being_polled.load(Ordering::Acquire) {
                spin_loop()
            }
            self.awaited_by.write().unwrap().undo();
            self.done.store(false, Ordering::Release);
            (*self.future.get()) = (self.task)(graph, self.clone())
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

    fn try_poll(self: &Arc<Self>) -> bool {
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

pub struct GraphBuilder<T: Task + ?Sized> {
    caller: usize,
    nodes: Rc<RefCell<Vec<Node>>>,
    marker: PhantomData<T>,
    is_leaf: bool,
    leafs: Arc<RwLock<UndoStack<usize>>>,
    awaits: Vec<usize>,
    //schedulers: Vec<Scheduler>,
}

impl<T: Task> GraphBuilder<T> {
    fn push(&mut self, node: Node) {
        self.nodes.borrow_mut().push(node)
    }

    fn next<U: Task>(&self) -> GraphBuilder<U> {
        GraphBuilder {
            caller: self.nodes.borrow().len() - 1,
            nodes: self.nodes.clone(),
            marker: PhantomData,
            is_leaf: true,
            leafs: self.leafs.clone(),
            awaits: vec![],
            //schedulers: vec![],
        }
    }

    pub fn spawn<U: Task>(&mut self, task: U, out: Option<*mut U::Output>) -> U::InitOutput {
        //if self.schedulers.is_empty() {
        //    self.schedulers.push(keep_local_schedular)
        //} else {
        //    self.schedulers.push(self.schedulers[self.caller])
        //}

        let len = self.nodes.borrow().len();
        self.push(Node::new::<U::Output>(len, out));
        self.nodes.borrow_mut()[len].name = U::name();
        if DEBUG {
            println!("{}", U::name())
        };
        let mut builder = self.next();
        let ret = task.init(&mut builder);

        if builder.is_leaf {
            println!("Pushed leaf");
            self.leafs.write().unwrap().push_extend(builder.caller);
        } else {
            if DEBUG {
                println!("NOT_LEAF: {} <- {:?}", self.caller, builder.awaits)
            };

            for awaits in &builder.awaits {
                self.nodes.borrow_mut()[*awaits]
                    .awaited_by
                    .get_mut()
                    .unwrap()
                    .push_extend(self.caller)
            }
        }
        ret
    }

    fn drain(self) -> Vec<Arc<Node>> {
        self.nodes
            .borrow_mut()
            .drain(..)
            .map(|n| {
                let mut q = n.awaited_by.write().unwrap();
                q.fix_capacity();
                q.checkpoint();
                drop(q);
                Arc::new(n)
            })
            .collect()
    }

    fn build(self) -> Arc<Graph> {
        let leafs = self.leafs.clone();
        leafs.write().unwrap().fix_capacity();
        leafs.write().unwrap().checkpoint();
        Graph::from_nodes(self.drain(), leafs)
    }

    //fn extends(self, graph: &Arc<Graph>) -> Arc<Graph> {
    //    graph.extend(self.drain())
    //}

    pub fn main(task: T) -> Self {
        let mut entry = Self {
            caller: 0,
            nodes: Rc::new(RefCell::new(vec![])),
            marker: PhantomData,
            leafs: Arc::new(RwLock::new(mpmc::Stack::new(0, 1).undoable())),
            is_leaf: true,
            awaits: vec![],
            //schedulers: vec![],
        };
        entry.spawn(task, None);
        entry
    }

    pub fn task(&mut self, task: AsyncFunction) {
        self.nodes.borrow_mut()[self.caller].task = task;
    }

    pub fn this_node(&mut self) -> Symbol<T::Output> {
        Symbol(self.caller, PhantomData)
    }

    pub fn lock_symbol<U>(&mut self, s: Symbol<U>) -> LockedSymbol<U> {
        self.is_leaf = false;
        self.awaits.push(s.0);
        LockedSymbol {
            returner: s.0,
            reader: self.caller,
            marker: PhantomData,
        }
    }

    pub fn set_mpi_instance(&mut self, mpi: i32) {
        self.nodes.borrow_mut()[self.caller].mpi_instance = mpi
    }

    // # Naiv distributed shcedular
    // we start with the 0-in-degree nodes
    // we have array where each element (usize) corresponds to one of those nodes
    // then evaluate children (taking array of indicies from parents).
    // Each child gets assigned to the index of the node that it had the most incomming edges from
    // repeat for children (taking array of the indicies assign to their parents)
    // also, dont count edges from parents that have already shared value with a given machine.
    //
    // the algorithm does not distribute nodes evenly, but it will minimize network communications
    // it also does not account for message size or computation cost
    pub fn scheduler(&mut self, scheduler: Scheduler) {
        //self.schedulers[self.caller] = scheduler
    }

    pub fn mutate_node(&mut self, f: impl Fn(&mut Node)) {
        f(&mut self.nodes.borrow_mut()[self.caller])
    }
}

pub struct Graph {
    // If this allocator is made reusable between graphs,
    // it would be safe to create a new graph inside an async block
    // and return a locked symbol from it. (also would require reusing the mpi universe)
    nodes: Vec<Arc<Node>>,
    _marker: PhantomPinned,
    pool: Arc<Pool>,
    leafs: Arc<RwLock<mpmc::UndoStack<usize>>>,
    // sub_graphs: Vec<GraphSpawner>
}
unsafe impl Sync for Graph {}
unsafe impl Send for Graph {}

impl Graph {
    pub fn from_nodes(
        nodes: Vec<Arc<Node>>,
        leafs: Arc<RwLock<mpmc::UndoStack<usize>>>,
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
                println!(
                    "{} awaited by {} on {}",
                    node.name,
                    reader.name,
                    self.mpi_instance()
                );
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
                Some(reader)
            })
            .collect()
    }

    fn assign_children_of<'a>(self: &'a Arc<Self>, node: &'a Arc<Node>) -> Option<&'a Arc<Node>> {
        let mut children = self.children(node);
        let continue_with = children.last();
        self.pool.assign(children.iter().copied());
        continue_with.copied()
    }

    pub fn compute(self: &Arc<Self>, node: usize) {
        let waker = Arc::new(NilWaker).into();
        let mut cx = Context::from_waker(&waker);

        let mut node = &self.nodes[node];

        loop {
            if DEBUG {
                println!("{} compputing {}", node.mpi_instance, node.name)
            };
            if node.done.load(Ordering::Acquire) {
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

                        if self.nodes[awaited]
                            .is_being_polled
                            .swap(true, Ordering::Acquire)
                        {
                            //node.awaited_by.read().unwrap().push(awaited, 0);
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

        println!(
            "Leafs: {:?}",
            self.leafs.write().unwrap().deref_mut().deref_mut()
        );

        self.pool
            .assign(self.leafs.write().unwrap().into_iter().filter_map(|n| {
                // FIXME: Initialy push children to leaf_node.awaiter
                //        (1 and 2 arent leaf nodes. Only X has no children)
                //        this is fine with pri-que, if leafs are just pushed with higher priority
                // TODO:  If there are devices left, after assigning leaf nodes.
                //        Then start assigning children.
                if DEBUG {
                    println!("LEAF: {n}")
                }
                if self.nodes[n].mpi_instance == self.mpi_instance() {
                    Some(&self.nodes[n])
                } else {
                    None
                }
            }));
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

// let ret = graph.new_buffer();
// let symbol: Symbol(usize) = graph.spawn(F(...), ret);
// let symbol: Symbol(ptr) = symbol.lock();
// graph.task(|_|async{ return symbol.await })

pub trait Task {
    type InitOutput;
    type Output: Serialize + Deserialize<'static> + Sync + 'static;
    fn init(self, graph: &mut GraphBuilder<Self>) -> Self::InitOutput;
    fn name() -> &'static str {
        type_name::<Self>()
    }
}

impl Task for () {
    type InitOutput = ();
    type Output = ();
    fn init(self, _: &mut GraphBuilder<Self>) -> Self::InitOutput {}
}

#[macro_export]
macro_rules! task {
    ($graph: ident, ($($cap: ident),* $(,)?), $f: expr) => {
        $graph.task(Box::new(move |_graph, _node| {
            $(let $cap = $cap.clone().own(&_graph);)*
            Box::pin(async move {
                let out: Self::Output = $f;
                unsafe{(*_node.output()) = out}
            })
        }))
    };
}

#[cfg(test)]
mod blur;
#[cfg(test)]
mod test {
    extern crate test;
    use std::sync::mpsc::channel;

    use crate::*;
    use test::black_box;
    use test::Bencher;

    /*
    struct GraphSpawner<T, O>{
        spawn: fn(parent_graph: &mut Graph, params: &T) -> Vec<Node>,
        params: Buffer,
        marker: PhantomData<O>
    }

    graph.sub_graph(source: Task<Output=T>) -> smth idk
    */
    //struct AnonymousTask<Args, O>(
    //    fn(&Arc<Graph>, Arc<Node>, Args) -> BoxFuture,
    //    Args,
    //    PhantomData<O>,
    //);
    //impl<Args: Sync + Send + 'static, O: Sync + Serialize + Deserialize<'static> + 'static> Task
    //    for AnonymousTask<Args, O>
    //{
    //    type InitOutput = Symbol<O>;
    //    type Output = O;

    //    fn init(self, graph: &mut GraphBuilder<Self>) -> Self::InitOutput {
    //        graph.task(Box::new(move |graph, node| {
    //            Box::pin(async move {
    //                let Self(task, args, _) = self;
    //                unsafe { *node.output() = task(graph, node.clone(), args) }
    //            })
    //        }));
    //        graph.this_node()
    //    }
    //}

    struct X;
    impl Task for X {
        type InitOutput = Symbol<f32>;
        type Output = f32;
        fn init(self, graph: &mut GraphBuilder<Self>) -> Self::InitOutput {
            //graph.set_mpi_instance(1);
            task!(graph, (), 2.);
            graph.this_node()
        }
    }

    struct F(Symbol<f32>);
    impl Task for F {
        // type Symbols = (Symbol<f32>);
        // graph.lock(self.0);
        // task!(graph, let (x) = node.own());
        type InitOutput = Symbol<f32>;
        type Output = f32;
        fn init(self, graph: &mut GraphBuilder<Self>) -> Self::InitOutput {
            let x = graph.lock_symbol(self.0);
            task!(graph, (x), unsafe { *(x.await.0) } * 3. + 4.);
            graph.this_node()
        }
    }

    struct Y;
    impl Task for Y {
        type InitOutput = ();
        type Output = ();
        fn init(self, graph: &mut GraphBuilder<Self>) -> Self::InitOutput {
            let x = graph.spawn(X, None);
            let x2 = graph.spawn(X, None);
            let f = graph.spawn(F(x), None);
            let f2 = graph.spawn(F(x2), None);
            let f = graph.lock_symbol(f);
            let f2 = graph.lock_symbol(f2);
            let x = graph.lock_symbol(x);
            task!(graph, (x, f, f2), {
                let y = unsafe { *f.await.0 };
                let y2 = unsafe { *f2.await.0 };
                let x = unsafe { *x.await.0 };
                println!("f({x}) = ({y}) = {y2}")
            });
        }
    }

    // if you just have a node for spawning a sub-graph,
    // that takes inputs from inside async block (ie not symbols) and returns a graph (Output=Graph),
    // you can call it on a new device that needs to spawn a task from that sub-graph.
    // combined with reusing stuff from parent graph.

    /*
    #[morphic]
    fn F(graph, x: Symbol<f32>) -> f32{
        let x = graph.own_symbol(x);
        task!(graph, x.await * 3. + 3);
    }
    */

    //#[test]
    //fn f() {
    //    let graph = black_box(Graph::new());
    //    graph.handle().spawn(Y);
    //    graph.realize();
    //}

    #[bench]
    fn f_of_x(b: &mut Bencher) {
        //let mut y = 0f32;
        let builder = GraphBuilder::main(Y);
        let graph = builder.build();
        let (net_events, mut net) = graph.init_net();
        let lock = std::sync::Mutex::new(());
        b.iter(|| {
            let lock = lock.lock();
            graph.spin_down();
            graph.realize(net_events.clone());
            net.run();
            //assert_eq!(y, 10.);
            drop(lock);
        });
        println!("Finishing test...");
        graph.kill(net.kill());
    }

    #[bench]
    fn f_of_x_simple(b: &mut Bencher) {
        b.iter(|| {
            fn f(x: f32) -> f32 {
                x * 3. + 4.
            }
            fn y() {
                let x = black_box(2.);
                let y = black_box(f(x));
                println!("f({x}) = {y}");
            }
            black_box(y());
        })
    }

    #[bench]
    fn send_recv(b: &mut Bencher) {
        b.iter(|| {
            let (send, recv) = channel();
            send.send(black_box(2usize)).unwrap();
            black_box(recv.recv().unwrap());
        });
    }

    #[bench]
    fn spawn_async(b: &mut Bencher) {
        b.iter(|| {
            (|n: u32| {
                // black_box(n + 2); // slow
                async move { black_box(n + 2) } // fast
            })(black_box(2))
        })
    }

    #[bench]
    fn empty_vec(b: &mut Bencher) {
        let mut tmp = vec![];
        b.iter(|| {
            for _ in 0..black_box(10) {
                // tmp.push(black_box(3u32)); // slow
                tmp.push(black_box(())); // fast
                black_box(tmp.len());
            }
        })
    }

    #[bench]
    fn index(b: &mut Bencher) {
        b.iter(|| {
            black_box([1, 2, 3][black_box(2)]);
        })
    }

    #[bench]
    fn atomic_load(b: &mut Bencher) {
        b.iter(|| {
            let a = black_box(AtomicBool::new(black_box(false)));
            black_box(a.load(Ordering::Acquire));
        })
    }

    #[bench]
    fn mull_add_1(bench: &mut Bencher) {
        let a = black_box([
            0f32, 1f32, 2f32, 3f32, 4f32, 5f32, 6f32, 7f32, 8f32, 9f32, 10f32, 0f32, 1f32, 2f32,
            3f32, 4f32, 5f32, 6f32, 7f32, 8f32, 9f32, 10f32, 0f32, 1f32, 2f32, 3f32, 4f32, 5f32,
            6f32, 7f32, 8f32, 9f32, 10f32, 0f32, 1f32, 2f32, 3f32, 4f32, 5f32, 6f32, 7f32, 8f32,
            9f32, 10f32,
        ]);
        let b = black_box([
            0f32, 1f32, 2f32, 3f32, 4f32, 5f32, 6f32, 7f32, 8f32, 9f32, 10f32, 0f32, 1f32, 2f32,
            3f32, 4f32, 5f32, 6f32, 7f32, 8f32, 9f32, 10f32, 0f32, 1f32, 2f32, 3f32, 4f32, 5f32,
            6f32, 7f32, 8f32, 9f32, 10f32, 0f32, 1f32, 2f32, 3f32, 4f32, 5f32, 6f32, 7f32, 8f32,
            9f32, 10f32,
        ]);
        let c = black_box([
            0f32, 1f32, 2f32, 3f32, 4f32, 5f32, 6f32, 7f32, 8f32, 9f32, 10f32, 0f32, 1f32, 2f32,
            3f32, 4f32, 5f32, 6f32, 7f32, 8f32, 9f32, 10f32, 0f32, 1f32, 2f32, 3f32, 4f32, 5f32,
            6f32, 7f32, 8f32, 9f32, 10f32, 0f32, 1f32, 2f32, 3f32, 4f32, 5f32, 6f32, 7f32, 8f32,
            9f32, 10f32,
        ]);
        let mut d = black_box([0f32; 11 * 3]);
        bench.iter(|| {
            for n in 0..10 {
                black_box(d[n] = a[n] * b[n] + c[n]);
            }
        })
    }
    #[bench]
    fn mull_add_2(bench: &mut Bencher) {
        let a = black_box([
            0f32, 1f32, 2f32, 3f32, 4f32, 5f32, 6f32, 7f32, 8f32, 9f32, 10f32, 0f32, 1f32, 2f32,
            3f32, 4f32, 5f32, 6f32, 7f32, 8f32, 9f32, 10f32, 0f32, 1f32, 2f32, 3f32, 4f32, 5f32,
            6f32, 7f32, 8f32, 9f32, 10f32, 0f32, 1f32, 2f32, 3f32, 4f32, 5f32, 6f32, 7f32, 8f32,
            9f32, 10f32,
        ]);
        let b = black_box([
            0f32, 1f32, 2f32, 3f32, 4f32, 5f32, 6f32, 7f32, 8f32, 9f32, 10f32, 0f32, 1f32, 2f32,
            3f32, 4f32, 5f32, 6f32, 7f32, 8f32, 9f32, 10f32, 0f32, 1f32, 2f32, 3f32, 4f32, 5f32,
            6f32, 7f32, 8f32, 9f32, 10f32, 0f32, 1f32, 2f32, 3f32, 4f32, 5f32, 6f32, 7f32, 8f32,
            9f32, 10f32,
        ]);
        let c = black_box([
            0f32, 1f32, 2f32, 3f32, 4f32, 5f32, 6f32, 7f32, 8f32, 9f32, 10f32, 0f32, 1f32, 2f32,
            3f32, 4f32, 5f32, 6f32, 7f32, 8f32, 9f32, 10f32, 0f32, 1f32, 2f32, 3f32, 4f32, 5f32,
            6f32, 7f32, 8f32, 9f32, 10f32, 0f32, 1f32, 2f32, 3f32, 4f32, 5f32, 6f32, 7f32, 8f32,
            9f32, 10f32,
        ]);
        let mut d = black_box([0f32; 11 * 3]);
        //let mut e = black_box([0f32; 11 * 3]);
        bench.iter(|| {
            for n in 0..10 {
                black_box(d[n] = a[n] * b[n]);
                black_box(d[n] += c[n]);
            }
            //for n in 0..10 {
            //    black_box(e[n] = d[n] + c[n]);
            //}
        })
    }
}
