#![cfg(test)]
#![feature(test)]
#![feature(new_uninit)]

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
use std::hint::spin_loop;
use std::marker::{PhantomData, PhantomPinned};
use std::mem::{transmute, MaybeUninit};
use std::pin::Pin;
use std::rc::Rc;
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

pub struct Reader<T>(pub *const T);
unsafe impl<T> Send for Reader<T> {}

impl<T: 'static> Future for OwnedSymbol<T> {
    type Output = Reader<T>;

    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        if self.returner.done.load(Ordering::Acquire) {
            self.reader.qued.store(-1, Ordering::Release);
            // TODO: Replace with downcast_unchecked_ref()
            unsafe {
                Poll::Ready(Reader(
                    (*self.returner.output.get())
                        .data
                        .downcast_ref()
                        .expect("Tried to poll with incorrect runtime type")
                        as *const _,
                ))
            }
        } else {
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

pub struct Node {
    name: &'static str,
    task: AsyncFunction,
    future: UnsafeCell<BoxFuture>,
    qued: AtomicIsize,
    awaited_by: Receiver<usize>,
    awaiter: Sender<usize>,
    this_node: usize,
    output: UnsafeCell<Buffer>,
    done: AtomicBool,
    is_being_polled: AtomicBool,
    mpi_instance: i32,
    net_events: UnsafeCell<Option<Sender<net::Event>>>,
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
        (self.se)(unsafe { transmute::<&dyn Any, (*const (), &())>(self.data.as_ref()).0 })
            .expect("Buffer serialisation failed")
    }

    pub fn deserialize(&mut self, data: Vec<u8>) {
        (self.de)(data, unsafe {
            transmute::<&dyn Any, (*mut (), &())>(self.data.as_ref()).0
        })
        .unwrap()
    }
}

pub struct GraphBuilder<T: Task + ?Sized> {
    caller: usize,
    nodes: Rc<RefCell<Vec<Node>>>,
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
        self.push(Node::new::<U::Output>(len));
        self.nodes.borrow_mut()[len].name = U::name();
        task.init(&mut self.next())
    }

    fn drain(self) -> Vec<Arc<Node>> {
        self.nodes.borrow_mut().drain(..).map(Arc::new).collect()
    }

    fn build(self) -> Arc<Graph> {
        Graph::from_nodes(self.drain())
    }

    fn extends(self, graph: &Arc<Graph>) -> Arc<Graph> {
        graph.extend(self.drain())
    }

    pub fn main(task: T) -> Self {
        let mut entry = Self {
            caller: 0,
            nodes: Rc::new(RefCell::new(vec![])),
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

    pub fn set_mpi_instance(&mut self, mpi: i32) {
        self.nodes.borrow_mut()[self.caller].mpi_instance = mpi
    }
}

pub struct Graph {
    nodes: Vec<Arc<Node>>,
    _marker: PhantomPinned,
    pool: Arc<Pool>,
    mpi_instance: i32,
}
unsafe impl Sync for Graph {}
unsafe impl Send for Graph {}

impl Graph {
    pub fn from_nodes(nodes: Vec<Arc<Node>>) -> Arc<Self> {
        Arc::new_cyclic(|graph| Graph {
            mpi_instance: 0,
            nodes,
            _marker: PhantomPinned,
            pool: Pool::new(graph.clone()),
        })
    }

    pub fn extend(self: &Arc<Self>, nodes: Vec<Arc<Node>>) -> Arc<Self> {
        Arc::new(Graph {
            mpi_instance: self.mpi_instance,
            nodes,
            _marker: PhantomPinned,
            pool: self.pool.clone(),
        })
    }

    pub fn compute(self: &Arc<Self>, mut node: usize) {
        let waker = Arc::new(NilWaker).into();
        let mut cx = Context::from_waker(&waker);

        loop {
            if self.nodes[node].mpi_instance != self.mpi_instance {
                self.nodes[node]
                    .net()
                    .send(net::Event::AwaitNode(self.nodes[node].clone()))
                    .unwrap();
                return;
            }
            if self.nodes[node].done.load(Ordering::Acquire) {
                self.pool.assign(
                    self.nodes[node]
                        .awaited_by
                        .try_iter()
                        .map(|i| self.nodes[i].clone()),
                );
                return;
            }

            match unsafe { Pin::new(&mut *self.nodes[node].future.get()) }.poll(&mut cx) {
                Poll::Ready(()) => {
                    self.nodes[node].done.store(true, Ordering::Release);

                    self.pool.assign(
                        self.nodes[node]
                            .awaited_by
                            .try_iter()
                            .map(|i| self.nodes[i].clone()),
                    );
                    self.nodes[node]
                        .is_being_polled
                        .store(false, Ordering::Release);
                    return;
                }
                Poll::Pending => {
                    let awaiting = self.nodes[node].qued.load(Ordering::Acquire);
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
        assert_eq!(
            Arc::strong_count(&self),
            1,
            "Cannot realize Graph if there exists other references to it"
        );

        let (net_events, mut network) = net::instantiate(self.clone());
        self.pool.init(0);
        for n in &self.nodes {
            n.use_net(Some(net_events.clone()));
            n.respawn(&self)
        }
        self.compute(0);
        while !self.nodes[0].done.load(Ordering::Acquire) {
            spin_loop()
        }
        network.run();

        self.pool.finish();
        match Arc::try_unwrap(self) {
            Ok(this) => this.pool.kill(),
            Err(s) => panic!("fuck {}", Arc::strong_count(&s)),
        }
    }

    pub fn print(&self) {
        for n in 0..self.nodes.len() {}
    }
}

struct NilWaker;
impl Wake for NilWaker {
    fn wake(self: Arc<Self>) {
        todo!("Wake me up inside!")
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
                println!("f({x}) = {y}");
            });
        }
    }

    #[bench]
    fn f_of_x(b: &mut Bencher) {
        b.iter(|| {
            let builder = GraphBuilder::main(Y);
            let graph = builder.build();
            graph.realize();
        });
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
