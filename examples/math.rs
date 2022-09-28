use metalmorphosis::{execute, Program};
use std::{future::Future, pin::Pin};

#[derive(Debug)]
enum TestProgram {
    F(f32),
    G(f32),
    H(f32),
}

impl Program for TestProgram {
    type Future = Pin<Box<dyn Future<Output = ()>>>;

    fn future(self, cx: &'static metalmorphosis::TaskNode<Self>) -> Self::Future {
        use TestProgram::*;
        match self {
            F(x) => Box::pin(async move {
                let h = cx.branch::<f32>(H(x)).await.unwrap();
                let g = cx.branch::<f32>(G(h)).await.unwrap();
                println!("f({x}) = {g}");
            }),
            G(x) => Box::pin(async move { unsafe { cx.output(x * 4.).unwrap() } }),
            H(x) => Box::pin(async move { unsafe { cx.output(x.powi(3)).unwrap() } }),
        }
    }
}

fn main() {
    execute(TestProgram::F(2.)).unwrap();
}

// # Chain rule specs
//
// y.derivative(x) = dy/dx
//
//  *(a, b) = a * b
//  *(a, b).derivative(a) = b
//  *(a, b).derivative(b) = a
//
//  +(a, b) = a + b
//  +(a, b).derivative() = 1
//
//  inv(a) = 1 / a
//  inv(a).derivative() = -1 / x^2
//
//  f(g(x)) => f'(g(x)) * g'(x)
//
//  var(x) = x
//  var(x).derivative(x) = 1
//  var(x).derivative(!x) = x
//
//  f(g(x), h(x)) => f'(g(x)) * g'(x) + f'(h(x)) * h'(x)
//
//  f(g(x), h(x)) => f(g(x)).derivative(g) * g(x).derivative(x)
//                   + f(h(x)).derivative(h) * h(x).derivative(x)
//
//  (a+b)*b*c => 1*b*c + (a+b) * c
//
// ## How to reference nodes in program
//
// TaskNode::branch should return a future that can contain an index
// to the nodes position in the task graph.
//
// It can then be passed to the children,
// and they can do stuff like calculate derivatives in respect to that node.
