//! <div align="center">
//! <h1> metalmorphosis </h1>
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
//! - [ ] Benchmark two-stage blur
//! - [ ] Distribute (OpenMPI?)
//!     - don't time awaits inside node
//!     - reusing output in node would confuse executor
//! - [ ] Benchmark distributed
//!
//! ## Extra
//! - Allocator reusablility for dynamic graphs
//! - const graphs
//! - time-complexity hints
//! - static types for futures
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

#![cfg(test)]
#![feature(test)]

//mod buffer;
mod error;

use error::{Error, Result};
use serde::{Deserialize, Serialize};

use std::future::Future;
use std::marker::{PhantomData, PhantomPinned};
use std::mem::transmute;
use std::pin::Pin;
use std::ptr::null_mut;
use std::sync::atomic::Ordering::SeqCst;
use std::sync::atomic::{AtomicBool, AtomicIsize};
use std::sync::Arc;
use std::task::{Context, Poll, Wake};

// Should be a ref to the buffer, instead of usize.
// Should impl Future with #[always_use]
// the bool is if it has already been locked. So its not locked in a loop. (attached)
#[derive(Clone, Copy)]
struct Symbol<T: ?Sized>(*mut Node, PhantomData<T>);
#[derive(Clone, Copy)]
struct OwnedSymbol<T: ?Sized>(*mut Node, *mut Node, PhantomData<T>);

impl<T: 'static> Future for OwnedSymbol<T> {
    type Output = &'static T;

    // It doesn't seem that this must_use does anythin :/
    #[must_use]
    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        unsafe {
            if (*self.0).done {
                let rc = (*self.0).rc.fetch_sub(1, SeqCst);
                if rc >= 0 {
                    return Poll::Ready(transmute((*self.0).output.data));
                }
                if rc == -1 {
                    // we can re-init and poll it directly!
                    // remember to set rc.
                }
                if rc < -1 {
                    // someone else is starting to poll it. Hopefully...
                    // it should be very unlikely to end up here,
                    // as rc should be set as soon as a node is re-inited.
                    panic!(
                        "rc < -1; when {} awaiting {}",
                        (*self.1).this_node,
                        (*self.0).this_node
                    );
                }
            } else {
                //let poll_here = !(*self.0).is_being_polled.swap(true, SeqCst);
                //if poll_here {
                //    if Pin::new(&mut (*self.0).future).poll(cx).is_ready() {
                //        (*self.0).done = true;
                //        return Poll::Ready(transmute((*self.0).output.data));
                //    }
                //    (*self.0).is_being_polled.swap(false, SeqCst);
                //}
                (*self.1).awaiting = (*self.0).this_node
            }
            Poll::Pending
        }
    }
}

// future could have a known type at compile time,
// if it is allocated on the graphs bump allocator,
// and Node provides a funtion pointer for polling and initializing it.
struct Node {
    task: Box<dyn Fn() -> Pin<Box<dyn Future<Output = ()>>>>,
    future: Pin<Box<dyn Future<Output = ()>>>,
    readers: Vec<usize>,
    rc: AtomicIsize,
    awaiting: usize,
    this_node: usize,
    output: Buffer,
    done: bool,
    mpi_rank: usize,
    is_being_polled: AtomicBool,
    // sub_graphs: Vec<Graph> // which could be pushed to at runtime.
    //                        // The graphs would need some way to identify themselfes,
    //                        // so they arent rebuilt for no reason.
}

impl Node {
    fn new(this_node: usize) -> Self {
        Node {
            task: Box::new(|| Box::pin(async {})),
            future: Box::pin(async {}),
            readers: vec![],
            rc: AtomicIsize::from(0),
            awaiting: 0,
            this_node,
            output: Buffer::new(),
            done: false,
            mpi_rank: 0,
            is_being_polled: AtomicBool::new(false),
        }
    }

    fn respawn(&mut self) {
        if self.output.data == null_mut() {
            panic!("Looks like you forgot to initialize a buffer")
        }
        self.rc = AtomicIsize::new(self.readers.len() as isize);
        self.future = (self.task)();
    }
}

struct Buffer {
    data: *mut (),
    de: fn(&'static [u8], &mut ()) -> Result<()>,
    se: fn(&()) -> Result<Vec<u8>>,
}

impl Buffer {
    fn new() -> Self {
        Self {
            data: std::ptr::null_mut(),
            de: |_, _| Ok(()),
            se: |_| Ok(vec![]),
        }
    }

    fn from<'a, T: Serialize + Deserialize<'a>>(data: &mut T) -> Self {
        Buffer {
            data: data as *mut _ as *mut (),
            de: |b, out| {
                *unsafe { transmute::<_, &mut T>(out) } = bincode::deserialize(b)?;
                Ok(())
            },
            se: |v| Ok(bincode::serialize::<T>(unsafe { transmute(v) })?),
        }
    }
}

struct Graph {
    // If this allocator is made reusable between graphs,
    // it would be safe to create a new graph inside an async block
    // and return a locked symbol from it. (also would require reusing the mpi universe)
    bump: bumpalo::Bump,
    nodes: Vec<Node>,
    _marker: PhantomPinned,
}

struct GraphHandle<'a, T: Task + ?Sized> {
    graph: &'a mut Graph,
    calling: usize,
    marker: PhantomData<T>,
}

impl<'a, T: Task> GraphHandle<'a, T> {
    fn spawn<U: Task>(&mut self, task: U) -> U::InitOutput {
        let id = self.calling;
        self.calling = self.graph.nodes.len();
        self.graph.nodes.push(Node::new(self.calling));
        let ret = task.init(unsafe { transmute::<&mut Self, _>(self) });
        self.calling = id;
        ret
    }

    // attaches edge to self.
    fn own_symbol<U>(&mut self, s: Symbol<U>) -> OwnedSymbol<U> {
        unsafe {
            (*s.0).readers.push(self.calling);
            (*s.0).rc = AtomicIsize::new((*s.0).readers.len() as isize);
        }
        // dont fetch add here. Instead set `node.rc` to `node.readers.len()` when calling `node.task`
        // s.0.rc.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        OwnedSymbol(s.0, &mut self.graph.nodes[self.calling] as *mut _, s.1)
    }

    fn output(&mut self) -> *mut T::Output
    where
        <T as Task>::Output: Deserialize<'static> + Serialize,
        // If this size bound is removed, the compiler complains about casting thin pointer to a fat one...
    {
        let mut ptr = self.graph.nodes[self.calling].output.data as *mut T::Output;
        if ptr == null_mut() {
            self.uninit_buffer();
            ptr = self.graph.nodes[self.calling].output.data as *mut T::Output;
        }
        ptr
    }

    fn use_output(&mut self, o: &mut T::Output)
    where
        <T as Task>::Output: Deserialize<'static> + Serialize,
    {
        self.graph.nodes[self.calling].output = Buffer::from(o);
    }

    fn task(&mut self, task: Box<dyn Fn() -> Pin<Box<dyn Future<Output = ()>>>>) {
        self.graph.nodes[self.calling].task = task;
        self.graph.nodes[self.calling].respawn();
    }

    fn this_node(&mut self) -> Symbol<T::Output> {
        Symbol(&mut self.graph.nodes[self.calling] as *mut _, PhantomData)
    }

    fn alloc<U>(&mut self, val: U) -> &mut U {
        self.graph.bump.alloc(val)
    }

    fn uninit_buffer(&mut self)
    where
        <T as Task>::Output: Deserialize<'static> + Serialize,
    {
        unsafe {
            let _self = &mut *(self as *mut Self);
            self.use_output(_self.alloc(std::mem::MaybeUninit::<T::Output>::uninit().assume_init()))
        }
    }
}

impl Graph {
    fn handle<'a>(&'a mut self) -> GraphHandle<'a, ()> {
        GraphHandle {
            graph: self,
            calling: 0,
            marker: PhantomData,
        }
    }

    fn new() -> Self {
        Graph {
            bump: bumpalo::Bump::new(),
            nodes: vec![],
            _marker: PhantomPinned,
        }
    }

    fn realize(&mut self) {
        let waker = Arc::new(NilWaker).into();
        let mut cx = Context::from_waker(&waker);
        let mut node = 0;

        loop {
            if self.nodes[node].done {
                panic!("Cycle!")
            }
            self.nodes[node].awaiting = 0;
            match Pin::new(&mut self.nodes[node].future).poll(&mut cx) {
                Poll::Ready(()) => {
                    if node == 0 {
                        break;
                    }
                    self.nodes[node].done = true;
                    // TODO: Don't just read from the 0th reader
                    node = self.nodes[node].readers[0];
                }
                Poll::Pending => {
                    let awaiting = self.nodes[node].awaiting;
                    if awaiting != 0 {
                        node = awaiting
                    } else {
                        panic!("Pending nothing?");
                    }
                }
            }
        }
    }

    fn print(&self) {
        for n in 0..self.nodes.len() {
            println!("{n} -> {:?}", self.nodes[n].readers);
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

trait Task {
    type InitOutput;
    type Output: ?Sized;
    fn init(self, graph: &mut GraphHandle<Self>) -> Self::InitOutput;
}

impl Task for () {
    type InitOutput = ();
    type Output = ();
    fn init(self, _: &mut GraphHandle<Self>) -> Self::InitOutput {}
}

#[macro_export]
macro_rules! task {
    ($graph: ident, $f: expr) => {
        let out = $graph.output();
        $graph.task(Box::new(move || {
            Box::pin(async move {
                let f = $f;
                unsafe { *out = f }
            })
        }));
    };
}

#[cfg(test)]
mod test {
    extern crate test;
    use test::black_box;
    use test::Bencher;

    use crate::*;
    struct X;
    impl Task for X {
        type InitOutput = Symbol<f32>;
        type Output = f32;
        fn init(self, graph: &mut GraphHandle<Self>) -> Self::InitOutput {
            task!(graph, 2.);
            graph.this_node()
        }
    }

    struct F(Symbol<f32>);
    impl Task for F {
        type InitOutput = Symbol<f32>;
        type Output = f32;
        fn init(self, graph: &mut GraphHandle<Self>) -> Self::InitOutput {
            let x = graph.own_symbol(self.0);
            task!(graph, x.await * 3. + 4.);
            graph.this_node()
        }
    }

    struct Y;
    impl Task for Y {
        type InitOutput = ();
        type Output = ();
        fn init(self, graph: &mut GraphHandle<Self>) -> Self::InitOutput {
            let x = graph.spawn(X);
            let f = graph.spawn(F(x));
            let f = graph.own_symbol(f);
            let x = graph.own_symbol(x);
            task!(graph, println!("f({}) = {}", x.await, f.await));
        }
    }

    #[test]
    fn f_of_x() {
        let mut graph = Graph::new();
        graph.handle().spawn(Y);
        //graph.print();
        graph.realize();
    }

    /*
    struct Blurr3<'a> {
        data: Symbol<[u8]>,
        width: usize,
        height: usize,
    }
    impl<'a> Task for Blurr3<'a> {
        type InitOutput = Symbol<[u8]>;
        type Output = [u8];

        fn init(self, graph: &mut Graph) -> Self::InitOutput {
            let (height, width) = (self.height, self.width);
            assert_eq!(width % 3, 0);
            assert_eq!(height % 3, 0);

            let mut stage2 = vec![];
            let ret = graph.get_buffer(); // should return ref to data in buffer, not the actual buffer.
            let buffer = graph.alloc(vec![0; height*width]);

            for row in 0..(height / 3) {
                let a = Buffer::from(&mut buffer[...]);
                let b = Buffer::from(&mut ret[...]);

                let stage1 = graph.spawn(RowBlur3(self.data, self.width), a);
                stage2.push(graph.spawn(ColBlur3(stage1, self.height), b);
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

            graph.task(|_| async {
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
            });
            Self::edge(graph)
        }
    }
    */

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
