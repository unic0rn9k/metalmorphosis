//! <div align="center">
//! <h1> metalmorphosis </h1>
//! </div>
//!
//! # Definitions
//! - Symbol: a type used to refer to a node,
//!   that can be bound to another node, returning a future to the output of a node.
//!   (it lets you specify edges in the computation graph)
//!
//! - Dealocks will be caused by:
//! `graph.attach_edge(Self::edge(graph));`,
//! `graph.spawn(F(Self::edge(graph)));`.
//!
//! # TODO
//! - [ ] type checking
//! - [ ] awaiting nodes (buffer stuff, etc)
//! - [ ] runnable (executor)
//! - [ ] Benchmark two-stage blur
//! - [ ] Distribute (OpenMPI?)
//!     - don't time awaits inside node
//!     - reusing output in node would confuse executor
//! - [ ] Benchmark distributed
//!
//! # Extra
//! - Allocator reusablility for dynamic graphs
// bunch of stuff: https://play.rust-lang.org/?version=stable&mode=debug&edition=2021&gist=778be5ba4d57087abc788b5901bd780d
// some dyn shit: https://play.rust-lang.org/?version=stable&mode=debug&edition=2021&code=use%20std%3A%3Aany%3A%3ATypeId%3B%0A%0Astruct%20Symbol%3CT%3E(usize%2C%20PhantomData%3CT%3E)%3B%0A%0Astruct%20Node%3COutput%3E%7B%0A%20%20%20%20this_node%3A%20usize%2C%0A%20%20%20%20readers%3A%20Vec%3Cusize%3E%2C%0A%20%20%20%20output%3A%20Output%2C%0A%7D%0A%0Atrait%20Trace%7B%0A%20%20%20%20fn%20this_node(%26self)%20-%3E%20usize%3B%0A%20%20%20%20fn%20readers(%26self)%20-%3E%20Vec%3Cusize%3E%3B%0A%20%20%20%20fn%20output_type(%26self)%20-%3E%20TypeId%3B%0A%20%20%20%20%0A%20%20%20%20fn%20read%3CT%3E(%26mut%20self%2C%20name%3A%20%26str)%20-%3E%20Symbol%3CT%3E%7B%0A%20%20%20%20%20%20%20%20todo!()%3B%0A%20%20%20%20%7D%0A%7D%0A%0Astruct%20Graph%7B%0A%20%20%20%20nodes%3A%20Vec%3CBox%3Cdyn%20Trace%3E%3E%2C%0A%20%20%20%20is_locked%3A%20bool%2C%20%2F%2F%20any%20nodes%20spawned%20after%20is%20lock%20is%20set%2C%20will%20not%20be%20distributable%0A%7D%0A%0Astruct%20MainNode(*mut%20Graph)%3B%0A%0A%2F*%0Afn%20main()%7B%0A%20%20%20%20Graph%3A%3Anew().main(%7Cm%3A%20MainNode%7C%7B%0A%20%20%20%20%20%20%20%20m.spawn(%22x%22%2C%20Literal(2.3)%2C%20%5B%5D)%3B%0A%20%20%20%20%20%20%20%20m.spawn(%22y%22%2C%20Y%2C%20%5B%22x%22%5D)%3B%0A%20%20%20%20%20%20%20%20m.subgraph(%22mm%22%2C%20matmul)%3B%0A%20%20%20%20%20%20%20%20%2F%2F%20%22mm%22%20can%20only%20be%20a%20matmul%20graph%20tho.%20Not%20necessary%20if%20you%20can%20read%20nodes%20that%20have%20not%20been%20spawned%20yet.%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20let%20y%20%3D%20m.read%3A%3A%3Cf32%3E(%22y%22)%3B%0A%20%20%20%20%20%20%20%20let%20x%20%3D%20m.read%3A%3A%3Cf32%3E(%22x%22)%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20async%20%7B%0A%20%20%20%20%20%20%20%20%20%20%20%20%2F%2F%20%60for%20n%20in%200..x.next().await%60%20cannot%20be%20concistently%20optimized%0A%20%20%20%20%20%20%20%20%20%20%20%20%2F%2F%20mby%3A%20%60executor.hint(ScalesWith(%7Cs%7C%20s%20*%20x))%60%0A%20%20%20%20%20%20%20%20%20%20%20%20for%20n%20in%200..10%7B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20let%20y%3A%20f32%20%3D%20y.next().await%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20let%20x%3A%20f32%20%3D%20x.next().await%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20println!(%22%7Bn%7D%3A%20f(%7Bx%7D)%20%3D%20%7By%7D%22)%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%2F%2F%20Here%20the%20graph%20of%20%22mm%22%20can%20vary%20based%20on%20arguments%20that%20are%20computed%20inside%20async%20block!%0A%20%20%20%20%20%20%20%20%20%20%20%20m.init(%22mm%22%2C%20(10%2C%2010))%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%2F%2F%20%5E%5E%20Serialize%20and%20send%20arguments%20for%20initializing%20%22mm%22%20to%20all%20devices.%0A%20%20%20%20%20%20%20%20%20%20%20%20%2F%2F%20Initializing%20graph%20needs%20to%20be%20pure.%0A%20%20%20%20%20%20%20%20%7D%0A%20%20%20%20%7D)%0A%7D*%2F%0A%0A%0Atrait%20Symbolize%3CT%3E%7B%0A%20%20%20%20fn%20symbol(%26self)%20-%3E%20Symbol%3CT%3E%3B%20%20%20%20%0A%7D%0A%0A%0Aimpl%3CT%3E%20Symbolize%3CT%3E%20for%20Node%3CT%3E%7B%0A%20%20%20%20fn%20symbol(%26self)%20-%3E%20Symbol%3CT%3E%7B%0A%20%20%20%20%20%20%20%20Symbol(self.this_node)%0A%20%20%20%20%7D%0A%7D%0A%0A

#![cfg(test)]
#![feature(test)]

//mod buffer;
mod buffer;
mod error;

use error::{Error, Result};

use std::future::Future;
use std::marker::{PhantomData, PhantomPinned};
use std::sync::atomic::AtomicUsize;

// Should be a ref to the buffer, instead of usize.
// Should impl Future with #[always_use]
// the bool is if it has already been locked. So its not locked in a loop. (attached)
#[derive(Clone, Copy)]
struct Symbol<T, const LOCK: bool = false>(usize, PhantomData<T>);

struct Node {
    task: fn(&mut Node) -> Box<dyn Future<Output = ()>>,
    // cant keep future here if it should be possible to iterate in parallel.
    future: Option<Box<dyn Future<Output = ()>>>,
    readers: Vec<usize>,
    rc: AtomicUsize, // Should be in buffer.
    this_node: usize,
    iteration: usize,
    buffer: (),
}

impl Node {
    fn new(this_node: usize) -> Self {
        Node {
            this_node,
            task: |_| Box::new(async {}),
            future: None,
            readers: vec![],
            rc: AtomicUsize::from(0),
            iteration: 0,
            buffer: (),
        }
    }

    fn ret(&mut self) {
        todo!()
    }
}

struct Graph {
    // If this allocator is made reusable between graphs,
    // it would be safe to create a new graph inside an async block
    // and return a locked symbol from it.
    bump: bumpalo::Bump,
    nodes: Vec<Node>,
    calling: usize,
    marker: PhantomPinned,
}

impl Graph {
    // attaches edge to self.
    fn attach_edge<T>(&mut self, s: Symbol<T>) -> Symbol<T, true> {
        self.nodes[s.0].readers.push(self.calling);
        // TODO: dont fetch add here. Instead set `node.rc` to `node.readers.len()` when calling `node.task`
        self.nodes[s.0]
            .rc
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Symbol(s.0, s.1)
    }

    // should take a buffer as arg, which will used when creating the new tasks symbol.
    fn spawn<T: Task>(&mut self, task: T) -> T::InitOutput {
        let id = self.calling;
        self.calling = self.nodes.len();
        self.nodes.push(Node::new(self.calling));
        let ret = task.init(self);
        self.calling = id;
        ret
    }

    fn new() -> Self {
        Graph {
            bump: bumpalo::Bump::new(),
            nodes: vec![],
            calling: 0,
            marker: PhantomPinned,
        }
    }

    //fn new_buffer(&mut self) ->
}

// let ret = graph.new_buffer();
// let symbol: Symbol(usize) = graph.spawn(F(...), ret);
// let symbol: Symbol(ptr) = symbol.lock();
// graph.task(|_|async{ return symbol.await })

trait Task {
    type InitOutput;
    type Output;
    fn init(self, graph: &mut Graph) -> Self::InitOutput;
    fn edge(graph: &mut Graph) -> Symbol<Self::Output> {
        Symbol(graph.calling, PhantomData)
    }
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
        fn init(self, graph: &mut Graph) -> Self::InitOutput {
            // graph.task(|_|async{2.});
            Self::edge(graph)
        }
    }

    struct F(Symbol<f32>);
    impl Task for F {
        type InitOutput = Symbol<f32>;
        type Output = f32;
        fn init(self, graph: &mut Graph) -> Self::InitOutput {
            let x = graph.attach_edge(self.0);
            // graph.task(|_| async { x.await * a + b });
            Self::edge(graph)
        }
    }

    /*
    struct Y(Return<f32>);
    impl Task for Y {
        type InitOutput = Symbol<f32>;
        type Output = f32;
        fn init(self, graph: &mut Graph) -> Self::InitOutput {
            let x = graph.spawn(X, graph.new_buffer());
            let f = graph.spawn(F(x), self.0);
            //Self::edge(graph)
            f
        }
    }*/

    #[test]
    fn basic() {}

    #[bench]
    fn spawn_async(b: &mut Bencher) {
        b.iter(|| {
            (|n: u32| {
                //black_box(n + 2);
                async move { black_box(n + 2) }
            })(black_box(2))
        })
    }
}
