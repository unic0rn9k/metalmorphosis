use std::{
    any::type_name,
    cell::RefCell,
    marker::PhantomData,
    rc::Rc,
    sync::{Arc, RwLock},
};

use serde::{Deserialize, Serialize};

use crate::{mpsc, AsyncFunction, Graph, LockedSymbol, Node, Symbol};

pub struct GraphBuilder<T: Task + ?Sized> {
    caller: usize,
    nodes: Rc<RefCell<Vec<Node>>>,
    marker: PhantomData<T>,
    is_leaf: bool,
    leafs: Arc<RwLock<mpsc::UndoStack<usize>>>,
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
        }
    }

    pub fn spawn<U: Task>(&mut self, task: U, out: Option<*mut U::Output>) -> U::InitOutput {
        let len = self.nodes.borrow().len();
        self.push(Node::new::<U::Output>(len, out));
        self.nodes.borrow_mut()[len].name = U::name();
        let mut builder = self.next();
        let ret = task.init(&mut builder);
        if builder.is_leaf {
            self.leafs.write().unwrap().push_extend(builder.caller)
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

    pub fn build(self) -> Arc<Graph> {
        let leafs = self.leafs.clone();
        leafs.write().unwrap().fix_capacity();
        leafs.write().unwrap().checkpoint();
        Graph::from_nodes(self.drain(), leafs)
    }

    pub fn main(task: T) -> Self {
        let mut entry = Self {
            caller: 0,
            nodes: Rc::new(RefCell::new(vec![])),
            marker: PhantomData,
            leafs: Arc::new(RwLock::new(mpsc::Stack::new(0, 1).undoable())),
            is_leaf: true,
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
        self.nodes.borrow_mut()[s.0]
            .awaited_by
            .get_mut()
            .unwrap()
            .push_extend(self.caller);
        LockedSymbol {
            returner: s.0,
            reader: self.caller,
            marker: PhantomData,
        }
    }

    pub fn set_mpi_instance(&mut self, mpi: i32) {
        self.nodes.borrow_mut()[self.caller].mpi_instance = mpi
    }

    pub fn mutate_node<U>(&mut self, s: Symbol<U>, f: impl Fn(&mut Node)) {
        f(&mut self.nodes.borrow_mut()[s.0])
    }

    pub fn mutate_this_node(&mut self, f: impl Fn(&mut Node)) {
        let s = self.this_node();
        self.mutate_node(s, f)
    }
}

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
mod test {
    extern crate test;
    use std::sync::mpsc::channel;

    use super::*;
    use test::black_box;
    use test::Bencher;

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

            graph.mutate_node(f2, |f| f.mpi_instance = 1);

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

    /*
    #[morphic]
    fn F(graph, x: Symbol<f32>) -> f32{
        let x = graph.own_symbol(x);
        task!(graph, x.await * 3. + 3);
    }
    */

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
            let a = black_box(std::sync::atomic::AtomicBool::new(black_box(false)));
            black_box(a.load(std::sync::atomic::Ordering::Acquire));
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
