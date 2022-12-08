//! # Definitions
//! - Symbol: a name used to refer to a node. A future to the output of a node
//!   This has a concept of scope,
//!   meaning a given symbol might not refer to the same value in all nodes.
//!
//! # TODO
//! - [ ] type checking
//! - [ ] awaiting nodes (buffer stuff, etc)
//! - [ ] runnable (executor)
//! - [ ] Benchmark two-stage blur
//! - [ ] Distribute (OpenMPI?)
//! - [ ] Benchmark distributed
//!
//! # Extra
//! - dynamic graphs: a node that might spawn an unknown amount of sub nodes
//! - Node array: a struct that points to a node and lets you index into the nodes children

#![feature(test)]

mod error;

use error::{Error, Result};
use std::{any::TypeId, collections::HashMap, future::Future, marker::PhantomData};

trait Task: Sized {
    type Output;

    fn comp(self, node: NodeMeta) -> Result<NodeMeta>;
    fn node(self, this_node: usize, graph: *mut Graph) -> Result<NodeMeta> {
        let node = node(this_node, graph);
        self.comp(node)
    }
    fn name(&self) -> String {
        std::any::type_name::<Self>().to_string()
    }
    fn named(self, name: &str) -> Renamed<Self> {
        Renamed(self, name.to_string())
    }
    fn execute(self) {
        // assume self is the main function/node, and run it.
        // Should return Result
    }
}

struct Renamed<T: Task>(T, String);

impl<T: Task> Task for Renamed<T> {
    type Output = T::Output;
    fn comp(self, node: NodeMeta) -> Result<NodeMeta> {
        self.0.comp(node)
    }
    fn name(&self) -> String {
        self.1
    }
}

#[derive(Default)]
struct Graph {
    nodes: Vec<NodeMeta>,
    //symbols: HashMap<String, Symbol>,
}

impl Graph {
    fn get(&self, s: Symbol) -> &NodeMeta {
        &self.nodes[s.0]
    }
    fn get_mut(&mut self, s: Symbol) -> &mut NodeMeta {
        &mut self.nodes[s.0]
    }
}

// TODO: impl drop, so reader is removed from source
#[derive(Clone, Copy)]
struct Symbol(usize);

#[derive(Clone)]
struct NodeMeta<Output> {
    graph: *mut Graph,
    this_node: usize,
    scope: HashMap<String, Symbol>,
    sources: Vec<Symbol>, // Who does this node read from?
    readers: Vec<Symbol>, // Who reads from this node?
    //spawn: Vec<Symbol>,
    //then: Vec<Symbol>
    task: fn() -> Box<dyn Future<Output = ()>>, // fn() -> impl Future
    output: Output, // TODO: The call-parent should be responsible for allocating an output buffer.
}

impl NodeMeta {
    fn reads(mut self, name: &str) -> Result<NodeMeta> {
        // TODO: This method needs to have a Output type for the source node specified,
        // so it can provide some sort of symbol that can be awaited into a concrete value.
        let source = self.get_symbol(name)?;
        let this = self.symbol();
        self.mut_graph().get_mut(source).readers.push(this);
        self.sources.push(source);
        Ok(self)
    }

    //fn read(&mut self) ->

    fn spawn<'a, N: Task>(
        mut self,
        node: N,
        inherits: impl IntoIterator<Item = &'a str>,
    ) -> Result<NodeMeta> {
        if self.scope.get(&node.name()).is_some() {
            // Maybe this shouldn't error? It's just shadowing...
            return Err(Error::NameCollision(node.name()));
        }
        let new_node = self.graph().nodes.len();
        let graph = self.graph;
        self.mut_graph().nodes.push(node.node(new_node, graph)?);

        let mut scope = HashMap::new();
        for s in inherits {
            // TODO: Remap symbol names to new names, to imitate arguments.
            scope.insert(s.to_string(), self.get_symbol(s)?);
        }
        self.mut_graph().nodes[new_node].scope = scope;

        Ok(self)
    }

    fn graph(&self) -> &Graph {
        unsafe { &*self.graph }
    }

    fn mut_graph(&mut self) -> &mut Graph {
        unsafe { &mut *self.graph }
    }

    fn symbol(&self) -> Symbol {
        Symbol(self.this_node)
    }

    fn get_symbol(&self, l: &str) -> Result<Symbol> {
        if let Some(ok) = self.scope.get(l) {
            Ok(*ok)
        } else {
            Err(Error::UnknownNode(l.to_string()))
        }
    }

    //fn task(mut self, task: fn() -> impl Future) -> Node {
    //    self
    //}

    /*
    fn task(task: impl Future<Output=()>) -> impl Future<Output=()>{
        loop{
            // await unfinished sources from previous iterations
            // (they may be abandoned if they have no readers left)
            // reset buffer
            task.await;
            // inform all sources, that have not been read, that they will not be read
            // poll 'then'
        }
    }*/
}

fn node(this_node: usize, graph: *mut Graph) -> NodeMeta {
    NodeMeta {
        graph,
        this_node,
        sources: vec![],
        readers: vec![],
        scope: HashMap::new(),
        task: || Box::new(async {}),
    }
}

fn main_node(node: impl Task) -> Graph {
    let mut graph = Graph::default();
    let _ = node.node(0, &mut graph);
    graph
}

struct X;
impl Task for X {
    type Output = u32;
    fn comp(self, node: NodeMeta) -> Result<NodeMeta> {
        // # node array:
        // node.spawn(X.repeat(10))
        // node.read([X.repeat(10)])
        // node.task(|[x]| async{ x[0].await })

        // node.reads([&F.get(X)])
        // main -> F -> X
        // main -> this
        //node.clone().reads(&X)?.spawn(X, [])?.spawn(X, [])?; // error. Cannot spawn X twice
        node.clone()
            .reads(&X::name())?
            .spawn(X, [])?
            .spawn(X.n::<2>(), [])?;
        Ok(node)
    }
}

/*
struct Then<A: Grrrraph, B: Grrrraph>(PhantomData<(A, B)>);
trait Grrrraph: Sized {
    type LastNode: CompNode;
    type Prev: Grrrraph;
    fn pop() -> Option<Self::Prev>;
    fn push<T: Grrrraph>() -> Then<Self, T> {
        Then(PhantomData)
    }
}

fn bruh(a: &dyn Grrrraph) {
    a::Prev;
}*/

trait TypeMap: Sized + 'static {
    fn _id<T: 'static>(count: usize) -> usize;
    fn id<T: 'static>() -> usize {
        Self::_id::<T>(0)
    }
    fn and<B>() -> TypeAnd<Self, B> {
        TypeAnd(PhantomData)
    }
}

// TODO: Reverse ordering for and chaining
struct TypeAnd<RHS: TypeMap, T>(PhantomData<(RHS, T)>);

impl<RHS: TypeMap, A: 'static> TypeMap for TypeAnd<RHS, A> {
    fn _id<B: 'static>(count: usize) -> usize {
        if TypeId::of::<A>() == TypeId::of::<B>() {
            count
        } else {
            RHS::_id::<B>(count + 1)
        }
    }
}

impl TypeMap for () {
    fn _id<T: 'static>(_: usize) -> usize {
        panic!("Type not in TypeMap")
    }
}

#[test]
fn type_shit() {
    assert!(TypeAnd::<TypeAnd<(), u32>, u8>::_id::<u8>(0) == 0);
    assert!(TypeAnd::<TypeAnd<(), u32>, u8>::_id::<u32>(0) == 1);
}

extern crate test;
#[bench]
fn type_shishle(b: &mut test::Bencher) {
    use test::black_box;
    b.iter(|| {
        assert!(black_box(TypeAnd::<TypeAnd<(), u32>, u8>::id::<u8>() == 0));
    });
}

// trait TypeMap = map type -> int
// type Args: TypeMap;
// args: Args = [symbols...];
// Symbol(usize, TypeId);
// for arg in args{
//      assert arg.symbol.typeid == arg::T.typeid
// }

// args: [T]
// spawn(F, [node]){
//      node::Output == args::T
//      F()
// }

// # Incongruity between concept of symbols between functional and graph approach

// A1 -> A2 -> A3
//       |
//       V
// B1 -> B2 -> B3
//
// MAIN = spawn(A1[...], A2[...], ...)

// F(a, b) = a+b
// X = 0..10
// Z = 0..10
// Y = F(X, Z)
// Y = spawn(F, [x, z])

// # Solution A
//
// node.spawn(F.node([args]), [inherited])
//
// This means i should go back to using RenamedNode, because `inherited` is resolved AOT,
// and args are resolved at compile-time
//
// args are resolved by index
// let (a, b) = node.args();
//
// inherited nodes are resolved by name
// let x = node.get("x")
//
// # Solution B
//
// Y = spawn(F, [x.rename("a"), z.rename("b")])

// bunch of stuff: https://play.rust-lang.org/?version=stable&mode=debug&edition=2021&gist=778be5ba4d57087abc788b5901bd780d
// some dyn shit: https://play.rust-lang.org/?version=stable&mode=debug&edition=2021&code=use%20std%3A%3Aany%3A%3ATypeId%3B%0A%0Astruct%20Symbol%3CT%3E(usize%2C%20PhantomData%3CT%3E)%3B%0A%0Astruct%20Node%3COutput%3E%7B%0A%20%20%20%20this_node%3A%20usize%2C%0A%20%20%20%20readers%3A%20Vec%3Cusize%3E%2C%0A%20%20%20%20output%3A%20Output%2C%0A%7D%0A%0Atrait%20Trace%7B%0A%20%20%20%20fn%20this_node(%26self)%20-%3E%20usize%3B%0A%20%20%20%20fn%20readers(%26self)%20-%3E%20Vec%3Cusize%3E%3B%0A%20%20%20%20fn%20output_type(%26self)%20-%3E%20TypeId%3B%0A%20%20%20%20%0A%20%20%20%20fn%20read%3CT%3E(%26mut%20self%2C%20name%3A%20%26str)%20-%3E%20Symbol%3CT%3E%7B%0A%20%20%20%20%20%20%20%20todo!()%3B%0A%20%20%20%20%7D%0A%7D%0A%0Astruct%20Graph%7B%0A%20%20%20%20nodes%3A%20Vec%3CBox%3Cdyn%20Trace%3E%3E%2C%0A%20%20%20%20is_locked%3A%20bool%2C%20%2F%2F%20any%20nodes%20spawned%20after%20is%20lock%20is%20set%2C%20will%20not%20be%20distributable%0A%7D%0A%0Astruct%20MainNode(*mut%20Graph)%3B%0A%0A%2F*%0Afn%20main()%7B%0A%20%20%20%20Graph%3A%3Anew().main(%7Cm%3A%20MainNode%7C%7B%0A%20%20%20%20%20%20%20%20m.spawn(%22x%22%2C%20Literal(2.3)%2C%20%5B%5D)%3B%0A%20%20%20%20%20%20%20%20m.spawn(%22y%22%2C%20Y%2C%20%5B%22x%22%5D)%3B%0A%20%20%20%20%20%20%20%20m.subgraph(%22mm%22%2C%20matmul)%3B%0A%20%20%20%20%20%20%20%20%2F%2F%20%22mm%22%20can%20only%20be%20a%20matmul%20graph%20tho.%20Not%20necessary%20if%20you%20can%20read%20nodes%20that%20have%20not%20been%20spawned%20yet.%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20let%20y%20%3D%20m.read%3A%3A%3Cf32%3E(%22y%22)%3B%0A%20%20%20%20%20%20%20%20let%20x%20%3D%20m.read%3A%3A%3Cf32%3E(%22x%22)%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20async%20%7B%0A%20%20%20%20%20%20%20%20%20%20%20%20%2F%2F%20%60for%20n%20in%200..x.next().await%60%20cannot%20be%20concistently%20optimized%0A%20%20%20%20%20%20%20%20%20%20%20%20%2F%2F%20mby%3A%20%60executor.hint(ScalesWith(%7Cs%7C%20s%20*%20x))%60%0A%20%20%20%20%20%20%20%20%20%20%20%20for%20n%20in%200..10%7B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20let%20y%3A%20f32%20%3D%20y.next().await%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20let%20x%3A%20f32%20%3D%20x.next().await%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20println!(%22%7Bn%7D%3A%20f(%7Bx%7D)%20%3D%20%7By%7D%22)%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%2F%2F%20Here%20the%20graph%20of%20%22mm%22%20can%20vary%20based%20on%20arguments%20that%20are%20computed%20inside%20async%20block!%0A%20%20%20%20%20%20%20%20%20%20%20%20m.init(%22mm%22%2C%20(10%2C%2010))%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%2F%2F%20%5E%5E%20Serialize%20and%20send%20arguments%20for%20initializing%20%22mm%22%20to%20all%20devices.%0A%20%20%20%20%20%20%20%20%20%20%20%20%2F%2F%20Initializing%20graph%20needs%20to%20be%20pure.%0A%20%20%20%20%20%20%20%20%7D%0A%20%20%20%20%7D)%0A%7D*%2F%0A%0A%0Atrait%20Symbolize%3CT%3E%7B%0A%20%20%20%20fn%20symbol(%26self)%20-%3E%20Symbol%3CT%3E%3B%20%20%20%20%0A%7D%0A%0A%0Aimpl%3CT%3E%20Symbolize%3CT%3E%20for%20Node%3CT%3E%7B%0A%20%20%20%20fn%20symbol(%26self)%20-%3E%20Symbol%3CT%3E%7B%0A%20%20%20%20%20%20%20%20Symbol(self.this_node)%0A%20%20%20%20%7D%0A%7D%0A%0A
