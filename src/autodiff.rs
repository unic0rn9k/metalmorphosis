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

use crate::{Program, TaskNode, Work};

trait Differentiable<'a>: Sized + std::fmt::Debug + Send + Sync + 'a {
    fn forward_future<T: Program<'a> + From<Node<'a, Self>>>(
        &'a self,
        task_handle: &'a TaskNode<'a, T>,
    ) -> Work<'a>;

    fn derivative_future<T: Program<'a> + From<Derivative<'a, Self>>>(
        &'a self,
        symbol: &'static str,
        task_handle: &'a TaskNode<'a, T>,
    ) -> Work<'a>;
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
            Node::Add(_, _) => todo!(),
            Node::Sub(_, _) => todo!(),
            Node::Mul(_, _) => todo!(),
            Node::Div(_, _) => todo!(),
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
}
