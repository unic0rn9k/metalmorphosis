#![feature(future_join)]
use std::pin::Pin;

use metalmorphosis::*;

fn main() {
    use serde_derive::{Deserialize, Serialize};
    use std::future::{join, Future};

    unsafe impl MorphicIO for TestData {
        const IS_COPY: bool = true;
    }

    #[derive(Debug, Serialize, Deserialize, PartialEq, Clone, Copy)]
    enum TestData {
        A,
        B,
    }

    #[derive(Serialize, Deserialize, Debug, Clone, Copy)]
    enum TestProgram {
        Main,
        A,
        B(u32),
        C,
    }

    impl Program for TestProgram {
        type Future = Pin<Box<dyn Future<Output = ()>>>;

        fn future(self, task_handle: &'static TaskNode<Self>) -> Self::Future {
            use TestProgram::*;
            match self {
                Main => Box::pin(async move {
                    println!("::start\n:");

                    let a = task_handle.branch::<u32>(A);
                    let b = task_handle.branch::<u32>(B(2));
                    let c = task_handle.branch::<TestData>(C);
                    let (a, b) = join!(a, b).await;

                    assert_eq!(a.unwrap() + b.unwrap(), 3);
                    assert_eq!(c.await.unwrap(), TestData::B);

                    println!("::end");
                }),

                A => Box::pin(async move { unsafe { task_handle.write_output(1).unwrap() } }),
                B(n) => Box::pin(async move { unsafe { task_handle.write_output(n).unwrap() } }),

                C => Box::pin(
                    async move { unsafe { task_handle.write_output(TestData::B).unwrap() } },
                ),
            }
        }
    }

    execute(TestProgram::Main).unwrap();
}
