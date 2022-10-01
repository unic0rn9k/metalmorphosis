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

use std::{future::Future, marker::PhantomData, ops::Mul};

use crate::{work, BoxFuture, Program, TaskNode, Work};

trait Differentiable: Sized + std::fmt::Debug + Send + Sync {
    fn forward_future<T: Program + From<Self>>(self, task_handle: &TaskNode<T>) -> BoxFuture;

    fn derivative_future<T: Program + From<Derivative<Self>>>(
        self,
        symbol: &'static str,
        task_handle: &TaskNode<T>,
    ) -> BoxFuture;

    fn derivative<'a>(&'a self, symbol: &'static str) -> Node<'a, Self> {
        Node::Imported(self)
    }
}

#[derive(Debug, Clone, Copy)]
enum Node<'a, T: Differentiable> {
    Add(&'a Self, &'a Self),
    Sub(&'a Self, &'a Self),
    Mul(&'a Self, &'a Self),
    Div(&'a Self, &'a Self),
    Value(f32),
    WithSymbol(&'static str, &'a Self),
    Imported(&'a T),
}

#[derive(Debug, Clone, Copy)]
struct Derivative<T: Differentiable>(T, &'static str);

impl<'a, D: Differentiable> Differentiable for Node<'a, D> {
    fn forward_future<T: Program + From<Self>>(self, task_handle: &TaskNode<T>) -> BoxFuture {
        todo!()
    }

    fn derivative_future<T: Program + From<Derivative<Self>>>(
        self,
        symbol: &'static str,
        task_handle: &TaskNode<T>,
    ) -> BoxFuture {
        todo!()
    }
}

impl<'a, T: Differentiable> Program for Node<'a, T> {
    fn future<U: Program + From<Self>>(self, task_handle: &TaskNode<U>) -> Work {
        work(self.forward_future(task_handle))
    }
}

impl<T: Differentiable> Program for Derivative<T> {
    fn future<U: Program + From<Self>>(self, task_handle: &TaskNode<U>) -> Work {
        work(self.0.derivative_future(self.1, task_handle))
    }
}

struct MulNode<LHS: Mul<RHS, Output = O>, RHS, O>(
    PhantomData<LHS>,
    PhantomData<RHS>,
    PhantomData<O>,
);

impl<LHS: Mul<RHS, Output = O>, RHS, O> MulNode<LHS, RHS, O> {
    fn mul(lhs: LHS, rhs: RHS, o: &mut O) {
        *o = lhs * rhs
    }
}

// # Devide and conquer!
//
// Build type safe graph with ops (eg: +-*/)
//
// Stram type graph as iter of Program, with information about edges, to the executor
//
// Run the whole damn thing
