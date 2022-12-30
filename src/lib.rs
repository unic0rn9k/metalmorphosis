//! <div align="center">
//! <img src="https://raw.githubusercontent.com/unic0rn9k/metalmorphosis/4th_refactor/logo.png" width="300"/>
//! </div>
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
//! - [ ] return Result everywhere
//! - [X] handle for graph with type information about the node calling it.
//!
//! - [X] Executor / schedular
//!     - Wakers? (wake me up inside)
//! - [X] multithreaded
//!     - join future that works with array of symbols
//! - [ ] Benchmark two-stage blur
//!
//! - [ ] Distribute (OpenMPI?)
//!     - don't time awaits inside node
//!     - reusing output in node would confuse executor
//! - [ ] Simplify redundant code-paths (Graph::compute does not need to check if node is done two places, before polling children)
//! - [ ] Benchmark distributed
//!     - if I'm in a crunch for time, mby just make a synthetic benchmark... `thread::sleep(Duration::from_millis(10))`
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

// bunch of stuff: https://play.rust-lang.org/?version=stable&mode=debug&edition=2021&gist=778be5ba4d57087abc788b5901bd780d
// some dyn shit: https://play.rust-lang.org/?version=stable&mode=debug&edition=2021&code=use%20std%3A%3Aany%3A%3ATypeId%3B%0A%0Astruct%20Symbol%3CT%3E(usize%2C%20PhantomData%3CT%3E)%3B%0A%0Astruct%20Node%3COutput%3E%7B%0A%20%20%20%20this_node%3A%20usize%2C%0A%20%20%20%20readers%3A%20Vec%3Cusize%3E%2C%0A%20%20%20%20output%3A%20Output%2C%0A%7D%0A%0Atrait%20Trace%7B%0A%20%20%20%20fn%20this_node(%26self)%20-%3E%20usize%3B%0A%20%20%20%20fn%20readers(%26self)%20-%3E%20Vec%3Cusize%3E%3B%0A%20%20%20%20fn%20output_type(%26self)%20-%3E%20TypeId%3B%0A%20%20%20%20%0A%20%20%20%20fn%20read%3CT%3E(%26mut%20self%2C%20name%3A%20%26str)%20-%3E%20Symbol%3CT%3E%7B%0A%20%20%20%20%20%20%20%20todo!()%3B%0A%20%20%20%20%7D%0A%7D%0A%0Astruct%20Graph%7B%0A%20%20%20%20nodes%3A%20Vec%3CBox%3Cdyn%20Trace%3E%3E%2C%0A%20%20%20%20is_locked%3A%20bool%2C%20%2F%2F%20any%20nodes%20spawned%20after%20is%20lock%20is%20set%2C%20will%20not%20be%20distributable%0A%7D%0A%0Astruct%20MainNode(*mut%20Graph)%3B%0A%0A%2F*%0Afn%20main()%7B%0A%20%20%20%20Graph%3A%3Anew().main(%7Cm%3A%20MainNode%7C%7B%0A%20%20%20%20%20%20%20%20m.spawn(%22x%22%2C%20Literal(2.3)%2C%20%5B%5D)%3B%0A%20%20%20%20%20%20%20%20m.spawn(%22y%22%2C%20Y%2C%20%5B%22x%22%5D)%3B%0A%20%20%20%20%20%20%20%20m.subgraph(%22mm%22%2C%20matmul)%3B%0A%20%20%20%20%20%20%20%20%2F%2F%20%22mm%22%20can%20only%20be%20a%20matmul%20graph%20tho.%20Not%20necessary%20if%20you%20can%20read%20nodes%20that%20have%20not%20been%20spawned%20yet.%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20let%20y%20%3D%20m.read%3A%3A%3Cf32%3E(%22y%22)%3B%0A%20%20%20%20%20%20%20%20let%20x%20%3D%20m.read%3A%3A%3Cf32%3E(%22x%22)%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20async%20%7B%0A%20%20%20%20%20%20%20%20%20%20%20%20%2F%2F%20%60for%20n%20in%200..x.next().await%60%20cannot%20be%20concistently%20optimized%0A%20%20%20%20%20%20%20%20%20%20%20%20%2F%2F%20mby%3A%20%60executor.hint(ScalesWith(%7Cs%7C%20s%20*%20x))%60%0A%20%20%20%20%20%20%20%20%20%20%20%20for%20n%20in%200..10%7B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20let%20y%3A%20f32%20%3D%20y.next().await%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20let%20x%3A%20f32%20%3D%20x.next().await%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20println!(%22%7Bn%7D%3A%20f(%7Bx%7D)%20%3D%20%7By%7D%22)%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%2F%2F%20Here%20the%20graph%20of%20%22mm%22%20can%20vary%20based%20on%20arguments%20that%20are%20computed%20inside%20async%20block!%0A%20%20%20%20%20%20%20%20%20%20%20%20m.init(%22mm%22%2C%20(10%2C%2010))%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%2F%2F%20%5E%5E%20Serialize%20and%20send%20arguments%20for%20initializing%20%22mm%22%20to%20all%20devices.%0A%20%20%20%20%20%20%20%20%20%20%20%20%2F%2F%20Initializing%20graph%20needs%20to%20be%20pure.%0A%20%20%20%20%20%20%20%20%7D%0A%20%20%20%20%7D)%0A%7D*%2F%0A%0A%0Atrait%20Symbolize%3CT%3E%7B%0A%20%20%20%20fn%20symbol(%26self)%20-%3E%20Symbol%3CT%3E%3B%20%20%20%20%0A%7D%0A%0A%0Aimpl%3CT%3E%20Symbolize%3CT%3E%20for%20Node%3CT%3E%7B%0A%20%20%20%20fn%20symbol(%26self)%20-%3E%20Symbol%3CT%3E%7B%0A%20%20%20%20%20%20%20%20Symbol(self.this_node)%0A%20%20%20%20%7D%0A%7D%0A%0A

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
mod net;
mod workpool;

use error::{Error, Result};
use serde::{Deserialize, Serialize};
use workpool::Pool;

pub use mpi::time;
use std::any::{type_name, Any};
use std::cell::{RefCell, UnsafeCell};
use std::future::Future;
use std::marker::{PhantomData, PhantomPinned};
use std::mem::transmute;
use std::pin::Pin;
use std::rc::Rc;
use std::sync::atomic::Ordering;
use std::sync::atomic::{AtomicBool, AtomicIsize};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::task::{Context, Poll, Wake};

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
        let que = graph.nodes[self.returner].awaiter.clone();
        //que.send(self.reader).unwrap();
        OwnedSymbol {
            returner: graph.nodes[self.returner].clone(),
            reader: graph.nodes[self.reader].clone(),
            que,
            marker: PhantomData,
        }
    }
}

#[derive(Clone)]
pub struct OwnedSymbol<T> {
    returner: Arc<Node>,
    reader: Arc<Node>,
    que: Sender<usize>,
    marker: PhantomData<T>,
}
unsafe impl<T> Send for OwnedSymbol<T> {}

pub struct Reader<T>(pub *const T);
unsafe impl<T> Send for Reader<T> {}

impl<T: 'static> Future for OwnedSymbol<T> {
    type Output = Reader<T>;

    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        println!(
            "polling: {:?} for {:?}",
            self.returner.this_node, self.reader.this_node,
        );
        if self.returner.done.load(Ordering::Acquire) {
            println!("task was done");
            self.reader.qued.store(-1, Ordering::Release);
            unsafe { Poll::Ready(Reader((*self.returner.output.get()).ptr() as *const T)) }
        } else {
            println!("task was pending {}", self.returner.this_node);
            if self.reader.qued.load(Ordering::Acquire) == self.returner.this_node as isize {
                return Poll::Pending;
            }
            self.reader
                .qued
                .store(self.returner.this_node as isize, Ordering::Release);
            self.que.send(self.reader.this_node).unwrap();
            Poll::Pending
        }
    }
}

pub type BoxFuture = Pin<Box<dyn Future<Output = ()> + Send>>;
pub type AsyncFunction = Box<dyn Fn(&Arc<Graph>, Arc<Node>) -> BoxFuture>;

// Node could be split into two structs. One for everything that needs unsafe interior mutability. And one for the other stuff.
pub struct Node {
    name: &'static str,
    task: AsyncFunction,
    future: UnsafeCell<BoxFuture>, // X
    qued: AtomicIsize,             // This is just a Waker...
    awaited_by: Receiver<usize>,   // X
    awaiter: Sender<usize>,        // X
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
    fn new<T: Serialize + Deserialize<'static> + Sync + 'static>(this_node: usize) -> Self {
        let (awaiter, awaited_by) = channel();
        Node {
            name: "NIL",
            task: Box::new(|_, _| Box::pin(async {})),
            future: UnsafeCell::new(Box::pin(async {})),
            qued: AtomicIsize::new(-1),
            awaited_by,
            awaiter,
            this_node,
            output: UnsafeCell::new(Buffer::new::<T>()),
            done: AtomicBool::new(false),
            is_being_polled: AtomicBool::new(false),
            mpi_instance: 0,
            net_events: UnsafeCell::new(None),
        }
    }

    fn respawn(self: &Arc<Self>, graph: &Arc<Graph>) {
        unsafe { (*self.future.get()) = (self.task)(graph, self.clone()) }
    }

    fn use_net(self: &Arc<Self>, net: Option<Sender<net::Event>>) {
        unsafe { (*self.net_events.get()) = net }
    }

    fn net(self: &Arc<Self>) -> Sender<net::Event> {
        unsafe { (*self.net_events.get()).clone().expect("Network not set") }
    }

    // If this was not in Node, then check associated with downcasting would not be required.
    fn output<T: 'static>(self: &Arc<Self>) -> *mut T {
        unsafe {
            (*self.output.get()).data.downcast_mut().unwrap_or_else(|| {
                panic!(
                    "Tried to get output with incorrect runtime type. Expected {}",
                    type_name::<T>()
                )
            })
        }
    }
}

pub struct Buffer {
    data: Box<dyn Any>,
    de: fn(Vec<u8>, *mut ()) -> Result<()>,
    se: fn(*const ()) -> Result<Vec<u8>>,
}

impl Buffer {
    pub fn new<T: Serialize + Deserialize<'static> + Sync + 'static>() -> Self {
        unsafe {
            Buffer {
                data: Box::<T>::new_uninit().assume_init(),
                de: |b, out| {
                    *(out as *mut T) = bincode::deserialize(transmute(&b[..]))?;
                    Ok(())
                },
                se: |v| Ok(bincode::serialize::<T>(&*(v as *const T))?),
            }
        }
    }

    pub fn serialize(&self) -> Vec<u8> {
        (self.se)(self.ptr()).expect("Buffer serialisation failed")
    }

    pub fn deserialize(&mut self, data: Vec<u8>) {
        (self.de)(data, self.mut_ptr()).expect("Buffer deserialization failed")
    }

    fn ptr(&self) -> *const () {
        unsafe { transmute::<&dyn Any, (*const (), &())>(self.data.as_ref()).0 }
    }
    fn mut_ptr(&mut self) -> *mut () {
        unsafe { transmute::<&mut dyn Any, (*mut (), &())>(self.data.as_mut()).0 }
    }
}

/*
struct GraphSpawner<T, O>{
    spawn: fn(parent_graph: &mut Graph, params: &T) -> Vec<Node>,
    params: Buffer,
    marker: PhantomData<O>
}

graph.sub_graph(source: Task<Output=T>) -> smth idk
*/

pub struct GraphBuilder<T: Task + ?Sized> {
    caller: usize,
    nodes: Rc<RefCell<Vec<Node>>>,
    marker: PhantomData<T>,
    is_leaf: bool,
    leaf_recv: Rc<Receiver<usize>>,
    leaf_send: Sender<usize>,
    awaits: Vec<usize>,
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
            leaf_recv: self.leaf_recv.clone(),
            leaf_send: self.leaf_send.clone(),
            awaits: vec![],
        }
    }

    pub fn spawn<U: Task>(&mut self, task: U) -> U::InitOutput {
        self.is_leaf = false;
        let len = self.nodes.borrow().len();
        self.push(Node::new::<U::Output>(len));
        self.nodes.borrow_mut()[len].name = U::name();
        println!("{}", U::name());
        let mut builder = self.next();
        let ret = task.init(&mut builder);

        if builder.is_leaf {
            self.leaf_send.send(builder.caller).unwrap();
        } else {
            println!("NOT_LEAF: {} <- {:?}", self.caller, builder.awaits);
            for awaits in &builder.awaits {
                self.nodes.borrow_mut()[*awaits]
                    .awaiter
                    .send(self.caller)
                    .unwrap();
            }
        }
        ret
    }

    fn drain(self) -> Vec<Arc<Node>> {
        self.nodes.borrow_mut().drain(..).map(Arc::new).collect()
    }

    fn build(self) -> Arc<Graph> {
        let leafs = self.leaf_recv.clone();
        Graph::from_nodes(self.drain(), leafs)
    }

    //fn extends(self, graph: &Arc<Graph>) -> Arc<Graph> {
    //    graph.extend(self.drain())
    //}

    pub fn main(task: T) -> Self {
        let (leaf_send, leaf_recv) = channel();
        let mut entry = Self {
            caller: 0,
            nodes: Rc::new(RefCell::new(vec![])),
            marker: PhantomData,
            leaf_recv: Rc::new(leaf_recv),
            leaf_send,
            is_leaf: true,
            awaits: vec![],
        };
        entry.spawn(task);
        entry
    }

    pub fn task(&mut self, task: AsyncFunction) {
        self.nodes.borrow_mut()[self.caller].task = task;
    }

    pub fn this_node(&mut self) -> Symbol<T::Output> {
        Symbol(self.caller, PhantomData)
    }

    pub fn lock_symbol<U>(&mut self, s: Symbol<U>) -> LockedSymbol<U> {
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
}

pub struct Graph {
    // If this allocator is made reusable between graphs,
    // it would be safe to create a new graph inside an async block
    // and return a locked symbol from it. (also would require reusing the mpi universe)
    nodes: Vec<Arc<Node>>,
    _marker: PhantomPinned,
    pool: Arc<Pool>,
    leafs: Rc<Receiver<usize>>,
    // sub_graphs: Vec<GraphSpawner>
}
unsafe impl Sync for Graph {}
unsafe impl Send for Graph {}

impl Graph {
    pub fn from_nodes(nodes: Vec<Arc<Node>>, leafs: Rc<Receiver<usize>>) -> Arc<Self> {
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

    // Should only be called from Pool::assign
    pub fn compute(self: &Arc<Self>, mut node: usize) {
        let waker = Arc::new(NilWaker).into();
        let mut cx = Context::from_waker(&waker);

        // We should return at some point with a Poll<()>

        loop {
            //if self.nodes[node].mpi_instance != self.mpi_instance {
            //    self.nodes[node]
            //        .net()
            //        .send(net::Event::AwaitNode(self.nodes[node].clone()))
            //        .unwrap();
            //    return;
            //}
            println!(
                "{} compputing {}",
                self.nodes[node].mpi_instance, self.nodes[node].name
            );
            if self.nodes[node].done.load(Ordering::Acquire) {
                println!(
                    "{} already done {}",
                    self.mpi_instance(),
                    self.nodes[node].name
                );
                self.pool
                    .assign(self.nodes[node].awaited_by.try_iter().filter_map(|i| {
                        let reader = self.nodes[i].clone();
                        if reader.mpi_instance == self.mpi_instance() {
                            return Some(reader);
                        }
                        //if !reader.is_being_polled.swap(true, Ordering::Acquire) {
                        reader
                            .net()
                            .send(net::Event::NodeDone {
                                awaited: node,
                                reader: reader.this_node,
                            })
                            .unwrap();
                        //    self.nodes[node].awaiter.send(reader.this_node).unwrap();
                        //    reader.is_being_polled.store(false, Ordering::Release);
                        //} else {
                        //    panic!("Unable to get ownership, for cross machine await")
                        //}
                        None
                    }));
                self.nodes[node]
                    .is_being_polled
                    .store(false, Ordering::Release);
                return;
            }

            if self.nodes[node].mpi_instance != self.mpi_instance() {
                return;
            }

            match unsafe { Pin::new(&mut *self.nodes[node].future.get()) }.poll(&mut cx) {
                Poll::Ready(()) => {
                    if node == 0 {
                        self.nodes[node].net().send(net::Event::Kill).unwrap()
                    }
                    self.nodes[node].done.store(true, Ordering::Release);
                    continue;
                }
                Poll::Pending => {
                    let awaited = self.nodes[node].qued.load(Ordering::Acquire);
                    if awaited >= 0 {
                        let awaited = awaited as usize;
                        if self.nodes[awaited].mpi_instance != self.mpi_instance() {
                            self.nodes[node]
                                .net()
                                .send(net::Event::AwaitNode {
                                    awaited,
                                    reader: node,
                                })
                                .unwrap();
                            return;
                        }
                        self.nodes[node]
                            .is_being_polled
                            .store(false, Ordering::Release);

                        node = awaited
                    } else {
                        panic!("Pending nothing?");
                    }
                }
            }
        }
    }
    // # Better scheduler
    // ## Init
    // all nodes contain sender to shared que of tasks to be polled. (why cant it just be all threads?)
    // leaf nodes are sendt to shared que.
    //
    // ## Realize
    // Nodes from shared que are polled.
    // when parent returns pending, child is sendt to que.
    // quen and networking in loop.

    pub fn realize(self: Arc<Self>) {
        // It might be easyest/ a good solution, to find a distribution of tasks when initializing the graf,
        // and then improving the distribution at runtime. This could be achived with 'execution masks'.
        // Then if we make the graf queriable, and so that user can mutate the distribution mask manually,
        // and let user read benchmark data (How long execution/distribution took on different devices)
        // metalmorphosis suddenly provides primitives!
        //
        // Use Reciever::iter to implement custom pre-schedulers as closures.
        // If the the graph kept track of nodes reselution at evaluation time, this could be done dynamically.
        //
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
        assert_eq!(
            Arc::strong_count(&self),
            1,
            "Cannot realize Graph if there exists other references to it"
        );

        let (net_events, network) = net::instantiate(self.clone());
        self.pool.init(network.rank());
        for n in &self.nodes {
            n.use_net(Some(net_events.clone()));
            n.respawn(&self)
        }
        self.pool.assign(self.leafs.iter().filter_map(|n| {
            // FIXME: Initialy push children to leaf_node.awaiter
            // TODO:  If there are devices left, after assigning leaf nodes.
            //        Then start assigning children.
            if self.nodes[n].mpi_instance == self.mpi_instance() {
                println!("LEAF: {n}");
                Some(self.nodes[n].clone())
            } else {
                None
            }
        }));

        let hold_on = network.run();
        self.pool.finish();
        drop(hold_on);
        match Arc::try_unwrap(self) {
            Ok(this) => this.pool.kill(),
            Err(s) => panic!("fuck {}", Arc::strong_count(&s)),
        }
    }

    pub fn print(&self) {
        for node in &self.nodes {
            //if node.is_being_polled.swap(true, Ordering::Acquire) {
            //    println!("{} -> {} is being polled", node.mpi_instance, node.name);
            //    return;
            //}
            let mut awaiters = vec![];
            for n in node.awaited_by.try_iter() {
                awaiters.push((n, self.nodes[n].name));
            }
            for (n, _) in &awaiters {
                node.awaiter.send(*n).unwrap();
            }
            println!(
                "{} -> {} awaited by {:?}",
                node.mpi_instance, node.name, awaiters
            );
            node.is_being_polled.store(false, Ordering::Release);
        }
    }
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
            $(let $cap = $cap.own(&_graph);)*
            Box::pin(async move {
                let out: Self::Output = $f;
                unsafe{(*_node.output()) = out}
            })
        }));
    };
}

#[cfg(test)]
mod test {
    extern crate test;
    use std::sync::mpsc::channel;

    use crate::*;
    use test::black_box;
    use test::Bencher;

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

    // # This does not need to be multithreaded...
    // metalmorphosis::test::Y::0 -> metalmorphosis::test::F::2
    // metalmorphosis::test::F::2 -> metalmorphosis::test::X::1
    // metalmorphosis::test::F::2 <- metalmorphosis::test::X::1
    // metalmorphosis::test::X::1 <- metalmorphosis::test::F::2
    struct X;
    impl Task for X {
        type InitOutput = Symbol<f32>;
        type Output = f32;
        fn init(self, graph: &mut GraphBuilder<Self>) -> Self::InitOutput {
            graph.set_mpi_instance(1);
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
            let x = graph.spawn(X);
            let f = graph.spawn(F(x));
            let f = graph.lock_symbol(f);
            let x = graph.lock_symbol(x);
            task!(graph, (x, f), {
                let y = unsafe { *f.await.0 };
                let x = unsafe { *x.await.0 };
                println!("f({x}) = ({y})")
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
        b.iter(|| {
            let builder = GraphBuilder::main(Y);
            let graph = builder.build();
            graph.realize();
        });
    }

    /*
    struct Blurrr {
        data: Symbol<[u8]>,
        width: usize,
        height: usize,
    }
    impl Task for Blurrr {
        type InitOutput = Symbol<Vec<u8>>;
        type Output = Vec<u8>;

        fn init(self, graph: &mut GraphHandle<Self>) -> Self::InitOutput {
            let (height, width) = (self.height, self.width);
            assert_eq!(width % 3, 0);
            assert_eq!(height % 3, 0);

            let mut stage2 = vec![0u8; width * height];
            let ret = graph.output(); // should return ref to data in buffer, not the actual buffer.
            let buffer = graph.alloc(vec![0; height * width]);

            for row in 0..(height / 3) {
                //let a = Buffer::from(&mut buffer[...]);
                //let b = Buffer::from(&mut ret[...]);

                //let stage1 = graph.spawn(RowBlur3(self.data, self.width), a);
                //stage2.push(graph.spawn(ColBlur3(stage1, self.height), b);

                // `stage1` and `self.data` do not have the same type.
                // `let stage0: Symbol<[u8]> = Symbol.map(|val: &[u8]| &val[...] );`
                // Symbol::map<T, U> convert a Symbol<T> to a Symbol<U>
                // if you just change the type, and not the data, it should be zero-cost
            }

            // executor should se that this node has multiple sources, and should then try to distribute the work.
            // it shouldn't assign nodes to threads, if the nodes have pending sources that are already being processed.
            //
            // if a node checks its reader while being polled, and the reader is being polled on another device,
            // it should be the thread of the reading node that terminates.
            //
            // Don't start randomly distributing tasks, start with the ones that dont have any sources!

            task! {graph,
                todo!()
                // 2 row blurs, then you can do 1 col blur, if the input is paddet.
                // then 1 row blur per col blur.
                //
                // RowBlurr[x,y] should write to stage1[x,y]
                // RowBlurr[x,y] should read from stage0[x,y]
                // RowBlurr[x,y] should read from stage0[x+1,y]
                // RowBlurr[x,y] should read from stage0[x-1,y]
                //
                // RowBlurr[x,y] should write to stage2[x,y]
                // ColBlurr[x,y] should read from stage1[x,y]
                // ColBlurr[x,y] should read from stage1[x,y+1]
                // ColBlurr[x,y] should read from stage1[x,y-1]
            };
            graph.this_node()
        }
    }
    */

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
