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
//! - [ ] Executor / schedular
//!     - Wakers? (wake me up inside)
//! - [ ] multithreaded
//!     - thread pool
//!     - join future that works with array of symbols
//! - [ ] Benchmark two-stage blur
//!
//! - [ ] Distribute (OpenMPI?)
//!     - don't time awaits inside node
//!     - reusing output in node would confuse executor
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
// - a node, to be polled by recipient

// Each device keeps track of last device to not do anything.
// (0 is no device)
// There is a global AtomicUsize with the last device to not do anything.
// When a device becomes available, it sets its counter to its own id,
// and swaps its internal counter with the global.
//
// when a device recieves new work, it swaps the counters back again.
//
// if the device is on another machine, use latency table to decide if it should actually distribute.

#![cfg(test)]
#![feature(test)]

mod error;
mod workpool;

use error::{Error, Result};
use serde::{Deserialize, Serialize};
use workpool::Pool;

use std::any::type_name;
use std::cell::{RefCell, UnsafeCell};
use std::future::Future;
use std::marker::{PhantomData, PhantomPinned};
use std::mem::transmute;
use std::pin::Pin;
use std::sync::atomic::Ordering;
use std::sync::atomic::{AtomicBool, AtomicIsize};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Weak};
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
        OwnedSymbol {
            returner: graph.nodes[self.returner].clone(),
            reader: graph.nodes[self.reader].clone(),
            que: graph.nodes[self.returner].awaiter.clone(),
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

impl<T: 'static> Future for OwnedSymbol<T> {
    type Output = ();

    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        println!(
            "polling: {:?} for {:?}",
            self.returner.this_node, self.reader.this_node,
        );
        if self.returner.done.load(Ordering::Acquire) {
            // TODO: respawn if strong_count is 1
            println!("task was done");
            self.reader.qued.store(-1, Ordering::SeqCst);
            Poll::Ready(())
        } else {
            println!("task was pending {}", self.returner.this_node);
            if self.reader.qued.load(Ordering::SeqCst) == self.returner.this_node as isize {
                return Poll::Pending;
            }
            self.reader
                .qued
                .store(self.returner.this_node as isize, Ordering::SeqCst);
            self.que.send(self.reader.this_node).unwrap();
            Poll::Pending
        }
    }
}

pub type BoxFuture = Pin<Box<dyn Future<Output = ()> + Send>>;
pub type AsyncFunction = Box<dyn Fn(Weak<Graph>) -> BoxFuture>;

pub struct Node {
    name: &'static str,
    task: AsyncFunction,
    future: UnsafeCell<BoxFuture>,
    qued: AtomicIsize,
    awaited_by: Receiver<usize>,
    awaiter: Sender<usize>,
    this_node: usize,
    output: Buffer,
    done: AtomicBool,
    is_being_polled: AtomicBool,
}

impl Node {
    fn new(this_node: usize) -> Self {
        let (awaiter, awaited_by) = channel();
        Node {
            name: "NIL",
            task: Box::new(|_| Box::pin(async {})),
            future: UnsafeCell::new(Box::pin(async {})),
            qued: AtomicIsize::new(-1),
            awaited_by,
            awaiter,
            this_node,
            output: Buffer::new(),
            done: AtomicBool::new(false),
            is_being_polled: AtomicBool::new(false),
        }
    }

    fn respawn(&mut self, graph: Weak<Graph>) {
        self.future = UnsafeCell::new((self.task)(graph))
    }
}

pub struct Buffer {
    data: (),
    de: fn(&'static [u8], &mut ()) -> Result<()>,
    se: fn(&()) -> Result<Vec<u8>>,
}

impl Buffer {
    pub fn new() -> Self {
        Self {
            data: (),
            de: |_, _| Ok(()),
            se: |_| Ok(vec![]),
        }
    }

    pub fn from<'a, T: Serialize + Deserialize<'a>>(data: &mut T) -> Self {
        Buffer {
            data: (),
            de: |b, out| {
                *unsafe { transmute::<_, &mut T>(out) } = bincode::deserialize(b)?;
                Ok(())
            },
            se: |v| Ok(bincode::serialize::<T>(unsafe { transmute(v) })?),
        }
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

pub struct Graph {
    // If this allocator is made reusable between graphs,
    // it would be safe to create a new graph inside an async block
    // and return a locked symbol from it. (also would require reusing the mpi universe)
    //bump: Pin<Arc<bumpalo::Bump>>,
    nodes: Vec<Arc<Node>>,
    _marker: PhantomPinned,
    pool: Arc<Pool>,
    // sub_graphs: Vec<GraphSpawner>
}
unsafe impl Sync for Graph {}
unsafe impl Send for Graph {}

pub struct GraphBuilder<T: Task + ?Sized> {
    caller: usize,
    nodes: Arc<RefCell<Vec<Node>>>,
    marker: PhantomData<T>,
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
        }
    }

    pub fn spawn<U: Task>(&mut self, task: U) -> U::InitOutput {
        let len = self.nodes.borrow().len();
        self.push(Node::new(len));
        self.nodes.borrow_mut()[len].name = U::name();
        println!("{}", U::name());
        // TODO: did the child call spawn? if not, add it to a list of nodes to poll initialiy
        task.init(&mut self.next())
    }

    fn build(self) -> Arc<Graph> {
        Arc::new_cyclic(|graph| Graph {
            //bump: Pin::new(Arc::new(bumpalo::Bump::new())),
            nodes: self
                .nodes
                .borrow_mut()
                .drain(..)
                .map(|mut node| {
                    node.respawn(graph.clone());
                    Arc::new(node)
                })
                .collect(),
            _marker: PhantomPinned,
            pool: Pool::new(graph.clone()),
        })
    }

    pub fn main(task: T) -> Self {
        let mut entry = Self {
            caller: 0,
            nodes: Arc::new(RefCell::new(vec![])),
            marker: PhantomData,
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

    pub fn lock_symbol<U>(&self, s: Symbol<U>) -> LockedSymbol<U> {
        LockedSymbol {
            returner: s.0,
            reader: self.caller,
            marker: PhantomData,
        }
    }
}

impl Graph {
    pub fn compute(self: &Arc<Self>, mut node: usize) {
        let waker = Arc::new(NilWaker).into();
        let mut cx = Context::from_waker(&waker);

        // We should return at some point with a Poll<()>

        loop {
            println!("compputing: {node}");
            if self.nodes[node].done.load(Ordering::Acquire) {
                self.pool.assign(self.nodes[node].awaited_by.try_iter());
                return;
            }

            // TODO: Move this check into WorkPool::assign
            if self.nodes[node]
                .is_being_polled
                .swap(true, Ordering::Acquire)
            {
                return;
            }

            match unsafe { Pin::new(&mut *self.nodes[node].future.get()) }.poll(&mut cx) {
                Poll::Ready(()) => {
                    self.nodes[node].done.store(true, Ordering::Release);

                    self.pool.assign(self.nodes[node].awaited_by.try_iter());
                    self.nodes[node]
                        .is_being_polled
                        .store(false, Ordering::Release);
                    return;
                }
                Poll::Pending => {
                    let awaiting = self.nodes[node].qued.load(Ordering::SeqCst);
                    self.nodes[node]
                        .is_being_polled
                        .store(false, Ordering::Release);

                    if awaiting >= 0 {
                        node = awaiting as usize
                    } else {
                        panic!("Pending nothing?");
                    }
                }
            }
        }
    }

    pub fn realize(self: Arc<Self>) {
        // Graph realization does a topological sort,
        // where the 'Graph' is just an in-degree array.
        // Rc is the count of outgoing edges tho...
        //
        // Forks are stupid. If we keep track of incomming edges,
        // we can do topological sort.
        // This way we can always run all nodes that have no unresolved incomming edges in parralel.
        //
        // Nodes might be able to do computation before all incomming edges have been satisfied.
        // Nodes will only pend if they are awaiting other nodes,
        // so this can be solved by just polling all nodes iteratively, once when graph is realized.
        //
        // Realize computation flow:
        // - Poll all nodes, while collecting que of nodes that have 0 unresolved incomming edges.
        // - Poll all nodes in que on workpool.
        // - Collect all unresloved nodes that have 0 incomming edges into que.
        // - if que is empty:
        //      - if there are unresolved nodes: Graph contained cycle
        //      - else program is done
        // - Try to poll from que again
        //      - any time we get to this point, it means we missed an opertunity to poll a child.
        //        this is bad for performance. Log it!
        // Worker computation flow:
        // - Poll node.
        // - imediatly poll children that where awaiting this node.
        // - if there are multiple children:
        //      - this is where i was going to do 'forks'
        //
        // # Possible solution to forks
        // Each node has a que of pending children.
        // pool has a global que of nodes that could not find workers.
        // pool has a count of unoccupied workers.
        // When node has been resolved: Loop over pending children:
        // - If there are workers left: assign child to worker.
        // - else push the child to the global pool que.
        //
        // If a node has an incorrect incomming edge,
        // either it, or the parent, should be added to executor que.
        //
        // TODO:
        // Nodes should still poll parents, if it awaits it, and its not already being polled.
        // Once a child awaits a parent, the worker will always be unoccupied afterwards anyway.
        //
        // Just to be sure. Print if there are still unfinished nodes left before killing.
        //
        // It might be easyest/ a good solution, to find a distribution of tasks when initializing the graf,
        // and then improving the distribution at runtime. This could be achived with 'execution masks'.
        // Then if we make the graf queriable, and so that user can mutate the distribution mask manually,
        // and let user read benchmark data (How long execution/distribution took on different devices)
        // metalmorphosis suddenly provides primitives!
        assert_eq!(
            Arc::strong_count(&self),
            1,
            "Cannot realize Graph if there exists other references to it"
        );
        // Again. Less cursed with UnsafeCell
        unsafe { self.pool.init() }

        self.compute(0);
        while !self.nodes[0].done.load(Ordering::SeqCst) {
            std::hint::spin_loop()
        }

        // TODO: Keep polling until 0 is done (on any mpi instance).
        //       maybe we handle networking here, until 0 is done?
        // pool.network();

        // Check that other mpi instances are done before killing
        // This might mean that one of the mpi instances needs to know when to kill, and signal apropriately
        self.pool.kill();
    }

    pub fn print(&self) {
        for n in 0..self.nodes.len() {
            println!("{n} has {:?} refs", Arc::strong_count(&self.nodes[n]));
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
    type Output;
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
        //let mut out = $graph.output();
        $graph.task(Box::new(move |_graph| {
            Box::pin(async move {
                let _graph = _graph.upgrade().unwrap();
                $(let $cap = $cap.own(&_graph);)*
                let f = $f;
                //*out = f;
                //black_box(out);
                black_box(f);
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
            task!(graph, (), 2.);
            graph.this_node()
        }
    }

    struct F(Symbol<f32>);
    impl Task for F {
        type InitOutput = Symbol<f32>;
        type Output = f32;
        fn init(self, graph: &mut GraphBuilder<Self>) -> Self::InitOutput {
            let x = graph.lock_symbol(self.0);
            task!(graph, (x), {
                black_box(x.await);
                1. * 3. + 4.
            });
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
                //let (x, y) = join!(x, f).await;
                println!("----- got passed x");
                //let y = f.await;
                //let x = x.await;
                black_box(x.await);
                black_box(f.await);
                println!("========= f(x) = y")
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
