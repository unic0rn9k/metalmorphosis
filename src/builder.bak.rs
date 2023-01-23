use std::{
    any::type_name,
    cell::RefCell,
    collections::HashMap,
    marker::PhantomData,
    rc::Rc,
    sync::{Arc, RwLock},
};

use serde::{Deserialize, Serialize};

use crate::{mpsc, AsyncFunction, Graph, LockedSymbol, Node, Symbol, DEBUG};

pub struct GraphBuilder<T: Task + ?Sized> {
    caller: usize,
    nodes: Rc<RefCell<Vec<Node>>>,
    marker: PhantomData<T>,
    is_leaf: bool,
    leafs: Arc<RwLock<mpsc::UndoStack<usize>>>,
    parent_first_mutations: Rc<RefCell<HashMap<usize, ParentFirstMutation>>>,
    in_degrees: Rc<RefCell<Vec<usize>>>,
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
            parent_first_mutations: self.parent_first_mutations.clone(),
            in_degrees: self.in_degrees.clone(),
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
        self.in_degrees.borrow_mut().push(0);

        if DEBUG {
            println!("{}", U::name())
        };

        let mut builder = self.next();
        task.init(&mut builder)
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

        let mut in_degrees = self.in_degrees.borrow_mut().clone();
        let mutations = self.parent_first_mutations.borrow_mut().clone();
        let mut nodes = self.drain();

        let mut zero_in_degree = vec![];
        for n in 0..in_degrees.len() {
            if in_degrees[n] == 0 {
                println!("{n} is a leaf");
                leafs.write().unwrap().push_extend(n);
                zero_in_degree.push(n)
            }
        }

        leafs.write().unwrap().fix_capacity();
        leafs.write().unwrap().checkpoint();

        //while !zero_in_degree.is_empty() {
        //    let n = if let Some(n) = zero_in_degree.pop() {
        //        n
        //    } else {
        //        break;
        //    };
        //    if in_degrees[n] == 0 {
        //        if let Some(f) = mutations.get(&n) {
        //            let children = nodes[n]
        //                .awaited_by
        //                .read()
        //                .unwrap()
        //                .clone()
        //                .into_iter()
        //                .map(|n| {
        //                    in_degrees[n] -= 1;
        //                    if in_degrees[n] == 0 {
        //                        zero_in_degree.push(n)
        //                    }
        //                    nodes[n].clone()
        //                })
        //                .collect();
        //            f(
        //                Arc::get_mut(&mut nodes[n]).expect("GraphBuilder::build : Cycle in graph"),
        //                children,
        //            )
        //        }
        //    }
        //}

        Graph::from_nodes(nodes, leafs)
    }

    //fn extends(self, graph: &Arc<Graph>) -> Arc<Graph> {
    //    graph.extend(self.drain())
    //}

    pub fn main(task: T) -> Self {
        let mut entry = Self {
            caller: 0,
            nodes: Rc::new(RefCell::new(vec![])),
            marker: PhantomData,
            leafs: Arc::new(RwLock::new(mpsc::Stack::new(0, 1).undoable())),
            is_leaf: true,
            parent_first_mutations: Rc::new(RefCell::new(HashMap::new())),
            in_degrees: Rc::new(RefCell::new(vec![])),
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
        self.in_degrees.borrow_mut()[self.caller] += 1;
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

    pub fn parent_first_mutation(&mut self, mutation: ParentFirstMutation) {
        if self
            .parent_first_mutations
            .borrow_mut()
            .insert(self.caller, mutation)
            .is_some()
        {
            panic!("parent_first_mutation overwrite")
        }
    }

    pub fn mutate_node(&mut self, f: impl Fn(&mut Node)) {
        f(&mut self.nodes.borrow_mut()[self.caller])
    }
}

pub type ParentFirstMutation = fn(&mut Node, Vec<Arc<Node>>);

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

/// The first argument is a vector containing all the parents of the node being scheduled.
/// The second argument is the amount of mpi instances.
/// The output should be the mpi instance that the task will be executed on.
/// The output must never be greater than the amount of mpi instances.
pub type Scheduler = fn(Vec<&Node>, usize) -> usize;

pub fn keep_local_schedular(_: Vec<&Node>, _: usize) -> usize {
    0
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
