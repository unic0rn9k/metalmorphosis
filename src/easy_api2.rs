use std::{
    cell::UnsafeCell,
    future::Future,
    ops::Deref,
    sync::{Arc, Mutex, MutexGuard},
};

use serde::{Deserialize, Serialize};

use crate::{workpool::Pool, BoxFuture, Executor, Graph, Node, NodeId};

pub struct BasicProgram(Vec<Mutex<BoxFuture>>);

impl BasicProgram {
    pub fn new() -> Self {
        //Self(UnsafeCell::new(vec![]))
        Self(vec![])
    }
}

impl Graph for BasicProgram {
    fn len(&self) -> usize {
        //unsafe { (*self.0.get()).len() }
        self.0.len()
    }

    fn task(&self, id: usize) -> Box<dyn std::ops::DerefMut<Target = BoxFuture> + '_> {
        Box::new(self.0[id].lock().unwrap())
    }
}

pub struct Task<F, U>(fn(NodeId, U) -> F);

impl<T: 'static, F: Future<Output = T> + Send, U: Send> Task<F, U> {
    pub fn future(&'static self, node: NodeId, arg: U) -> BoxFuture {
        Box::pin(async move { unsafe { *node.output() = (self.0)(node.clone(), arg).await } })
    }
}

//fn task<F, U>(task: fn(NodeId, U) -> F) -> Task{Task}

impl BasicProgram {
    fn new_task<
        'a,
        F: Future<Output = T> + Send + 'static,
        T: Sync + Deserialize<'static> + Serialize + 'static,
        U: Send + 'static,
    >(
        &mut self,
        task: &'static Task<F, U>,
        args: U,
    ) -> NodeId {
        let node = Node::new::<T>(self.len()).commit();
        let ret = node.clone();
        //self.0.get_mut().push(task.future(node.clone(), args));
        self.0.push(Mutex::new(task.future(node.clone(), args)));
        ret
    }

    fn new_executor(self) -> Arc<Executor> {
        unsafe { Executor::new(self) }
    }
}

mod tests {
    use std::{hint::black_box, sync::atomic::Ordering};

    use crate::{dummy_net as net, mpmc::Stack};

    use super::*;
    use test::Bencher;

    extern crate test;

    #[bench]
    fn bruh(b: &mut Bencher) {
        let mut program = BasicProgram::new();

        let x_task = &Task(|_, _| async { 2f32 });
        let y_task =
            &Task(|this, x| async { 2. * unsafe { *this.edge_from::<f32>(x).await.0 } + 3. });

        let x = program.new_task(x_task, ());
        let y = program.new_task(y_task, x.clone());

        x.checkpoint();
        y.checkpoint();

        println!("LEN: {}", program.len());
        let exe = program.new_executor();
        let (net_events, mut net) = net::instantiate(exe.clone());
        unsafe {
            x.use_net(Some(net_events.clone()));
            y.use_net(Some(net_events.clone()));
        }

        b.iter(|| {
            exe.pool.paused.store(false, Ordering::SeqCst);
            exe.realize(&[x.clone(), y.clone(), x.clone(), y.clone()]);
            net.run();
            while !x.done.load(Ordering::SeqCst) || !y.done.load(Ordering::SeqCst) {}
            println!("threads: {}", exe.pool.live_threads());
            println!(
                "x:{},y:{}",
                x.done.load(Ordering::SeqCst),
                y.done.load(Ordering::SeqCst)
            );

            y.respawn();
            x.respawn();

            exe.pool.paused.store(true, Ordering::SeqCst);
            println!("= PAUSED =");

            //exe.leftovers = Stack::new(100, 2);

            let x_: f32 = unsafe { *x.output() };
            let y_: f32 = unsafe { *y.output() };
            assert_eq!(x_, 2.);
            assert_eq!(y_, 7.);
            println!("f({x_}) = {y_}\n=== RESETTING ===");

            unsafe {
                *x.output() = 0f32;
                *y.output() = 0f32;
            }
            **exe.graph.task(0) = x_task.future(x.clone(), ());
            **exe.graph.task(1) = y_task.future(y.clone(), x.clone());
        });

        println!("=== ALL DONE ===");
        exe.kill(net.kill());
    }

    fn table(img: &[f32], dim: &[usize]) {
        let brightness = [".", ":", ";", "!", "|", "?", "&", "=", "%", "#", "@"];

        for y in 0..dim[1] {
            for x in 0..dim[0] {
                let n = img[x + y * dim[0]];
                //print!(" {:.3}", n); // numeric
                print!("{}", brightness[(n * 20.).min(10.) as usize]); // visual
            }
            println!();
        }
        println!();
    }

    #[bench]
    fn basic_blur(b: &mut Bencher) {
        fn blur_x(img: &[f32], output: &mut [f32], dim: &[usize; 2]) {
            let img = |x: isize, y: isize| {
                if x < 0 || x >= dim[0] as isize || y < 0 || y >= dim[1] as isize {
                    0f32
                } else {
                    img[x as usize + y as usize * dim[0]]
                }
            };

            let [x, y] = dim;
            for y in 0..*y as isize {
                for x in 0..*x as isize {
                    let p = (img(x, y + 1) + img(x, y - 1) + img(x, y)) / 3.;
                    // output[x as usize + y as usize * dim[0]] = p; // non-transposed output
                    output[y as usize + x as usize * dim[1]] = p; // transposed output
                }
            }
        }

        const DIM: [usize; 2] = [160, 30];

        //#[rustfmt::skip]
        //let input = black_box([
        //  0f32,0f32,0f32,0f32,0f32,0f32,
        //  0f32,0f32,0f32,0f32,0f32,0f32,
        //  0f32,0f32,1f32,1f32,0f32,0f32,
        //  0f32,0f32,0f32,1f32,0f32,0f32,
        //  0f32,0f32,0f32,0f32,0f32,0f32,
        //  0f32,0f32,0f32,0f32,0f32,0f32,
        //]);

        let mut input = black_box([0f32; DIM[0] * DIM[1]]);
        for x in 0..DIM[0] {
            let k = DIM[1] as f32 / 2.;
            let y = (x as f32 * 0.1).sin() * k + k;
            input[x + y as usize * DIM[0]] = 1.;
        }

        let mut output = black_box([0f32; DIM[0] * DIM[1]]);
        let mut horizontal = black_box([0f32; DIM[0] * DIM[1]]);

        b.iter(|| {
            let trans = [DIM[1], DIM[0]];
            blur_x(&input, &mut horizontal, &DIM);
            //blur_x(&input, &mut horizontal, &trans);
            blur_x(&horizontal, &mut output, &trans);
            //blur_x(&horizontal, &mut output, &DIM);

            black_box(output);
        });

        table(&input, &DIM);
        table(&output, &DIM);
    }
}
//struct Blur {
//    input: *const [f32],
//    x_blur: Vec<f32>,
//    result: Vec<f32>,
//    width: usize,
//    height: usize,
//    tasks: Vec<BoxFuture>,
//}
//
//impl Blur {
//    fn new(img: &[f32], width: usize) -> Self {
//        let height = img.len() / width;
//
//        let blur_x = Node::new::<Vec<f32>>(0);
//        let blur_y = Node::new::<Vec<f32>>(1);
//
//        let mut tasks = vec![];
//
//        Self {
//            input: img as *const _,
//            x_blur: vec![0.; width * height],
//            result: vec![0.; width * height],
//            width,
//            height,
//            tasks,
//        }
//    }
//}
//
//unsafe impl Graph for Blur {}
//
//#[bench]
//fn blur(b: &mut Bencher) {
//    let width = 10;
//    let height = 10;
//    let image = [032; width * height];
//}
