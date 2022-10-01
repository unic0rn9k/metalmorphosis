#![feature(future_join, pin_macro)]
use std::pin::{pin, Pin};

use metalmorphosis::{execute, executor::Executor, work, MorphicIO, Program, TaskNode, Work};

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

    impl<'a> Program<'a> for TestProgram {
        fn future<T: Program<'a> + From<Self>>(self, task_handle: &'a TaskNode<'a, T>) -> Work<'a> {
            use TestProgram::*;
            work(async move {
                match self {
                    Main => {
                        println!("::start\n:");

                        let a = task_handle.branch::<u32>(A);
                        let b = task_handle.branch::<u32>(B(2));
                        let c = task_handle.branch::<TestData>(C);
                        let (a, b) = join!(a, b).await;

                        assert_eq!(a.unwrap() + b.unwrap(), 3);
                        assert_eq!(c.await.unwrap(), TestData::B);

                        println!("::end");
                    }

                    A => unsafe { task_handle.output(1).unwrap() },
                    B(n) => unsafe { task_handle.output(n).unwrap() },

                    C => unsafe { task_handle.output(TestData::B).unwrap() },
                }
            })
        }
    }

    execute(TestProgram::Main).unwrap()
}
