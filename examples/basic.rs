#![feature(future_join, pin_macro)]
use std::pin::{pin, Pin};

use metalmorphosis::{
    error::Result, execute, executor::Executor, work, MorphicIO, TaskHandle, TaskNode, Work,
};

fn main() {
    use serde_derive::{Deserialize, Serialize};
    use std::future::{join, Future};

    unsafe impl MorphicIO<'_> for TestData {
        const IS_COPY: bool = true;
    }

    #[derive(Debug, Serialize, Deserialize, PartialEq, Clone, Copy)]
    enum TestData {
        A,
        B,
    }

    fn a<'a>(task_handle: TaskHandle<'a, u8>) -> Work<'a> {
        work(async move { task_handle.output(2).unwrap() })
    }

    fn b<'a>(task_handle: TaskHandle<'a, u8>) -> Work<'a> {
        work(async move { task_handle.output(1).unwrap() })
    }

    fn c<'a>(task_handle: TaskHandle<'a, TestData>) -> Work<'a> {
        work(async move { task_handle.output(TestData::B).unwrap() })
    }

    fn morphic_main<'a>(task_handle: TaskHandle<'a, ()>) -> Work<'a> {
        work(async move {
            println!("::start");

            let a = task_handle.branch(a);
            println!("Can branch");
            let b = task_handle.branch(b);
            let c = task_handle.branch(c);
            let (a, b) = join!(a, b).await;
            println!("Can await");

            assert_eq!(a.unwrap() + b.unwrap(), 3);
            assert_eq!(c.await.unwrap(), TestData::B);

            println!("::end");

            /*
                            B(n) => unsafe { task_handle.output(n).unwrap() },

                            C => unsafe { task_handle.output(TestData::B).unwrap() },
            */
        })
    }

    execute(morphic_main)
}

//#[test]
//fn test() {
//    main()
//}
