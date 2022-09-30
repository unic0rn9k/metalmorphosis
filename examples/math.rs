use levitate::Float;
use metalmorphosis::{Program, TaskNode, _async, execute};
use std::{future::Future, pin::Pin};

pub struct Symbol<const NAME: char>;

impl<const A: char, const B: char> PartialEq<Symbol<B>> for Symbol<A> {
    #[inline(always)]
    fn eq(&self, _: &Symbol<B>) -> bool {
        A == B
    }
}

#[derive(Debug)]
enum TestProgram {
    F(f32),
    G(f32),
    H(f32),
    DF(f32, char),
}

/*
impl Program for TestProgram {
    type Future = Pin<Box<dyn Future<Output = ()>>>;

    // Is it ok to have concurrent imutable access to cx?
    // For sure need to make a wrapper struct.
    fn future(self, cx: &'static metalmorphosis::TaskNode<Self>) -> Self::Future {
        use TestProgram::*;
        match self {
            F(x) => Box::pin(async move {
                let h = cx.branch::<f32>(H(x)).await.unwrap();
                let g = cx.branch::<f32>(G(h)).await.unwrap();
                println!("f({x}) = {g}");
            }),
            G(x) => _async! { unsafe { cx.output(x * 4.).unwrap() } },
            H(x) => _async! { unsafe { cx.output(x.powi(3)).unwrap() } },
            DF(x, sym) => Box::pin(async move {
                //if par ==   x  => g'(h(x)) * h'(x)
                //if par == h(x) => h'(x)
                //if par == g(x) => 1
                let d: f32 = match sym {
                    'x' => G(H(x)).derivative('x') * H(x).derivative('x'),
                    'h' => H(x).derivative('x'),
                    'g' => 1.,
                    _ => panic!("Symbol {sym:?} not found"),
                };
            }),
        }
    }
}
*/

fn main() {
    //execute(TestProgram::F(2.)).unwrap();
}
