use std::{future::Future, sync::Arc};

use serde::{Deserialize, Serialize};

use crate::{workpool::Pool, BoxFuture, Executor, Graph, Node, NodeId};

pub struct BasicProgram(Vec<BoxFuture>);

impl BasicProgram {
    pub fn new() -> Self {
        Self(vec![])
    }
}

unsafe impl Graph for BasicProgram {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn task(&self, id: usize) -> &BoxFuture {
        &self.0[id]
    }

    fn task_mut(&mut self, id: usize) -> &mut BoxFuture {
        &mut self.0[id]
    }

    //fn init(&mut self, exe: Arc<crate::Graph>) {
    //    todo!()
    //}
}

impl BasicProgram {
    fn new_task<
        'a,
        F: Future<Output = T> + Send + 'static,
        T: Sync + Deserialize<'static> + Serialize + 'static,
        U: Send + 'static,
    >(
        &mut self,
        task: fn(NodeId, U) -> F,
        args: U,
    ) -> NodeId {
        let mut node = Node::new::<T>(self.0.len());
        let node = node.commit();
        let ret = node.clone();
        self.0.push(Box::pin(async move {
            unsafe { *node.output() = task(node.clone(), args).await }
        }));
        ret
    }

    fn new_executor(self) -> Arc<Executor> {
        unsafe { Executor::new(self) }
    }
}

mod tests {
    use crate::dummy_net as net;

    use super::*;
    use test::Bencher;

    extern crate test;

    #[bench]
    fn bruh(b: &mut Bencher) {
        let mut program = BasicProgram::new();

        let x = program.new_task(|_, _| async { 2f32 }, ());

        let y = program.new_task(
            |this, x| async { 2. * unsafe { *this.edge_from::<f32>(x).await.0 } + 3. },
            x.clone(),
        );

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
            exe.realize(&[x.clone(), y.clone()]);
            net.run();

            x.respawn();
            y.respawn();

            let x: f32 = unsafe { *x.output() };
            let y: f32 = unsafe { *y.output() };

            assert_eq!(x, 2.);
            assert_eq!(y, 7.);
            println!("f({x}) = {y}");
        });

        exe.kill(net.kill());
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
