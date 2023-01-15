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

    pub fn spawn<U: Task>(&mut self, task: U) -> U::InitOutput {
        //if self.schedulers.is_empty() {
        //    self.schedulers.push(keep_local_schedular)
        //} else {
        //    self.schedulers.push(self.schedulers[self.caller])
        //}

        let len = self.nodes.borrow().len();
        self.push(Node::new::<U::Output>(len));
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

    fn drain(self) -> Vec<Node> {
        self.nodes
            .borrow_mut()
            .drain(..)
            .map(|n| {
                let mut q = n.awaited_by.write().unwrap();
                q.fix_capacity();
                q.checkpoint();
                drop(q);
                n
            })
            .collect()
    }

    fn build(self) -> Arc<Executor> {
        let leafs = self.leafs.clone();
        leafs.write().unwrap().fix_capacity();
        leafs.write().unwrap().checkpoint();
        Executor::from_nodes(self.drain(), leafs)
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
        self.is_leaf = false;
        self.awaits.push(s.0);
        LockedSymbol {
            awaited: s.0,
            awaiter: self.caller,
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
        //$graph.task(Box::new(move |_graph, _node| {
        //    $(let $cap = $cap.own(&_graph);)*
        //    Box::pin(async move {
        //        let out: Self::Output = $f;
        //        unsafe{(*_node.output()) = out}
        //    })
        //}));
        todo!()
    };
}

/* TESTS & BENCHES */

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
    //    fn(&Arc<Graph>, &Node, Args) -> BoxFuture,
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
        let builder = GraphBuilder::main(Y);
        let graph = builder.build();
        let (net_events, mut net) = graph.init_net();
        let lock = std::sync::Mutex::new(());
        b.iter(|| {
            let lock = lock.lock();
            graph.spin_down();
            graph.realize(net_events.clone());
            net.run();
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
                //println!("f({x}) = {y}");
            }
            black_box(y());
        })
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
