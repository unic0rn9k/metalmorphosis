#![feature(future_join, pin_macro)]
use metalmorphosis::{execute, executor::halt_once, work, MorphicIO, TaskHandle, Work};

fn main() {
    use serde_derive::{Deserialize, Serialize};
    use std::future::join;

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

    // This really should not work 😬
    fn shouldnt_work<'a>(handle: TaskHandle<'a, &'a u8>) -> Work<'a> {
        work(async move { task_handle.output(&1).unwrap() })
        // It should work if the TaskNode was kept alive...
        // like with this:
        // ```rust
        // loop{
        //     halt_once().await
        // }
        // ```
    }

    fn morphic_main<'a>(task_handle: TaskHandle<'a, ()>) -> Work<'a> {
        // Maybe do something so only preloaded branches can be distributed.
        let preloaded = task_handle.branch(a);

        work(async move {
            println!("::start");

            for _ in 0..10 {
                let _ = preloaded.await;
            }

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

    fn mpi_main(){
        let a = buffer;
        if rank == 0{
            a.read()
        }
        if rank == 1{
            a.write()
        }
    }

    execute(morphic_main)
}

//#[test]
//fn test() {
//    main()
//}
