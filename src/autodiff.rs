//! # Chain rule specs
//!
//! y.derivative(x) = dy/dx
//!
//!  *(a, b) = a * b
//!  *(a, b).derivative(a) = b
//!  *(a, b).derivative(b) = a
//!
//!  +(a, b) = a + b
//!  +(a, b).derivative() = 1
//!
//!  inv(a) = 1 / a
//!  inv(a).derivative() = -1 / x^2
//!
//!  f(g(x)) => f'(g(x)) * g'(x)
//!
//!  var(x) = x
//!  var(x).derivative(x) = 1
//!  var(x).derivative(!x) = x
//!
//!  f(g(x), h(x)) => f'(g(x)) * g'(x) + f'(h(x)) * h'(x)
//!
//!  f(g(x), h(x)) => f(g(x)).derivative(g) * g(x).derivative(x)
//!                   + f(h(x)).derivative(h) * h(x).derivative(x)
//!
//!  (a+b)*b*c => 1*b*c + (a+b) * c
//!
//! ## How to reference nodes in program
//!
//! TaskNode::branch should return a future that can contain an index
//! to the nodes position in the task graph.
//!
//! It can then be passed to the children,
//! and they can do stuff like calculate derivatives in respect to that node.
//!
//! ## Relevant literature
//! <https://thenumb.at/Autodiff/>
//!
//! # Goal check list
//! - [ ] Basic float autodiff
//! - [ ] Matrix autodiff
//! - [ ] Caching computed values in forward pass, for use in differentiation
//! - [ ] Pre-allocation of needed memory for entire graph
//! - [ ] Parallele matrix multiplication

use crate::{Result, TaskNode};
use std::marker::PhantomData;

/*
trait Differentiable<'a>: Sized + std::fmt::Debug + Send + Sync + 'a {
    type Output: Differentiable<'a>;

    fn forward_future<T: Program<'a> + From<Node<'a, Self>>>(
        &'a self,
        task_handle: &'a TaskNode<'a, T>,
    ) -> Work<'a>;

    fn derivative_future<T: Program<'a> + From<Derivative<'a, Self>>>(
        &'a self,
        symbol: &'static str,
        task_handle: &'a TaskNode<'a, T>,
    ) -> Work<'a>;

    fn add(&'a self, other: &'a Self) -> Self::Output;
}

#[derive(Debug, Clone, Copy)]
enum Node<'a, T: Differentiable<'a>> {
    Add(&'a Self, &'a Self),
    Sub(&'a Self, &'a Self),
    Mul(&'a Self, &'a Self),
    Div(&'a Self, &'a Self),
    WithSymbol(&'static str, &'a Self),
    Imported(&'a T),
}

impl<'a, T: Differentiable<'a>> Node<'a, T> {
    fn derivative(&'a self, symbol: &'static str) -> Derivative<'a, T> {
        Derivative(self, symbol)
    }
}

#[derive(Debug, Clone, Copy)]
struct Derivative<'a, T: Differentiable<'a>>(&'a Node<'a, T>, &'static str);

impl<'a, T: Differentiable<'a>> Program<'a> for Node<'a, T> {
    fn future<U: Program<'a> + From<Self>>(self, task_handle: &'a TaskNode<'a, U>) -> Work<'a> {
        match self {
            Node::Add(lhs, rhs) => unsafe {
                todo!()
                //let lhs = task_handle.branch::<LHS>(lhs);
                //let rhs = task_handle.branch::<RHS>(rhs);
                //let (lhs, rhs) = join!(lhs, rhs).await;
                //task_handle
                //    .output::<O>(lhs.unwrap() + rhs.unwrap())
                //    .unwrap();
            },
            Node::Sub(lhs, rhs) => todo!(),
            Node::Mul(lhs, rhs) => todo!(),
            Node::Div(lhs, rhs) => todo!(),
            Node::WithSymbol(_, _) => todo!(),
            Node::Imported(inner) => inner.forward_future(task_handle),
        }
    }
}

impl<'a, T: Differentiable<'a>> Program<'a> for Derivative<'a, T> {
    fn future<U: Program<'a> + From<Self>>(self, task_handle: &'a TaskNode<'a, U>) -> Work<'a> {
        match self.0 {
            Node::Add(_, _) => todo!(),
            Node::Sub(_, _) => todo!(),
            Node::Mul(_, _) => todo!(),
            Node::Div(_, _) => todo!(),
            Node::WithSymbol(_, _) => todo!(),
            Node::Imported(inner) => inner.derivative_future(self.1, task_handle),
        }
    }
}*/

// Hvis jeg kan lave en graph, der returner `TaskNode<'a, T>` kan jeg anvende `task_handle.attach_tree().await.unwrap()`

// Hvis alle child nodes ligger i den højre ende af graphen som et array, vil man mby kunne lave noget smart med executoren når den skal tjekke tasks.

// TODO: A way to define type safe math and convert it to a sequence of nodes, with zero overhead
trait Node<'a>: Sized {
    type Child;
    // Promise that `child.future(task_handle).await` will return `Self::Output` to parent through `task_handle`
    type Output;

    type Derivative: Node<'a, Output = Self::Output>;

    fn sub_nodes() -> usize;
    // fn children() -> usize; // ?

    fn eval(this_node: &'a TaskNode<'a>) -> Result<'a, &'a TaskNode<'a>>;

    fn collect_into(parent: &'a TaskNode<'a>, task_buffer: &mut [TaskNode<'a>]) -> Result<'a, ()> {
        // task_buffer[0 or this_node](Self::eval(tash_handle.orphan(..)))
        // LHS::future(task_handle.orphan(..), &mut task_buffer[1..])
        // RHS::future(task_handle.orphan(..), &mut task_buffer[LHS::sub_nodes()..])
        todo!()
    }

    fn collect(parent: &'a TaskNode<'a>) -> Result<'a, Vec<TaskNode<'a>>> {
        let mut buffer: Vec<TaskNode<'a>> = Vec::with_capacity(Self::sub_nodes());
        Self::collect_into(parent, &mut buffer)?;
        Ok(buffer)
    }

    fn derivative() -> Self::Derivative;
}

struct Mulf32<'a, LHS: Node<'a, Output = f32>, RHS: Node<'a, Output = f32>> {
    children: (LHS, RHS),
    phantom_data: PhantomData<&'a ()>,
}

struct Addf32<'a, LHS: Node<'a, Output = f32>, RHS: Node<'a, Output = f32>> {
    children: (LHS, RHS),
    phantom_data: PhantomData<&'a ()>,
}

impl<'a, LHS: Node<'a, Output = f32>, RHS: Node<'a, Output = f32>> Node<'a>
    for Mulf32<'a, LHS, RHS>
{
    type Child = (LHS, RHS);
    type Output = f32;
    type Derivative =
        Addf32<'a, Mulf32<'a, LHS::Derivative, RHS>, Mulf32<'a, RHS::Derivative, LHS>>;

    fn sub_nodes() -> usize {
        RHS::sub_nodes() + LHS::sub_nodes()
    }

    fn eval(this_node: &'a TaskNode<'a>) -> Result<'a, &'a TaskNode<'a>> {
        // Here we should read directly from the child node,
        // as all nodes where added at the same time,
        // this node will only get polled once the child has written a value anyway.
        todo!()
    }

    fn derivative() -> Self::Derivative {
        todo!()
    }
}

impl<'a, LHS: Node<'a, Output = f32>, RHS: Node<'a, Output = f32>> Node<'a>
    for Addf32<'a, LHS, RHS>
{
    type Child = (LHS, RHS);
    type Output = f32;
    type Derivative = Addf32<'a, LHS::Derivative, RHS::Derivative>;

    fn sub_nodes() -> usize {
        RHS::sub_nodes() + LHS::sub_nodes()
    }

    fn eval(this_node: &'a TaskNode<'a>) -> Result<'a, &'a TaskNode<'a>> {
        todo!()
    }

    fn derivative() -> Self::Derivative {
        todo!()
    }
}
