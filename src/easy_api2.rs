use std::{future::Future, sync::Arc};

use serde::{Deserialize, Serialize};

use crate::{BoxFuture, Graph, Node};

pub struct BasicProgram(Vec<BoxFuture>);

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

// Should return Symbol<T>
impl BasicProgram {
    fn push<'a, T: Sync + Deserialize<'static> + Serialize + 'static>(
        &mut self,
        task: impl Future<Output = T> + Send + 'static,
    ) -> Arc<Node> {
        let node = Arc::new(Node::new::<T>(self.0.len()));
        let ret = node.clone();
        self.0.push(Box::pin(
            async move { unsafe { *node.output() = task.await } },
        ));
        ret
    }
}

#[bench]
fn bruh(b: &mut Bencher) {
    let mut program = BasicProgram::new();

    let x = program.push(async { 2f32 });
    let y = program.push(async move { 2. * x.await + 3. });

    let exe = program.default_executor();

    b.iter(|| {
        let x = exe.eval(x);
        let y = exe.eval(y);

        println!("f({x}) = {y}");
    });
}

struct Blur {
    input: *const [f32],
    x_blur: Vec<f32>,
    result: Vec<f32>,
    width: usize,
    height: usize,
    tasks: Vec<BoxFuture>,
}

impl Blur {
    fn new(img: &[f32], width: usize) -> Self {
        let height = img.len() / width;

        let blur_x = Node::new::<Vec<f32>>(0);
        let blur_y = Node::new::<Vec<f32>>(1);

        let mut tasks = vec![];

        Self {
            input: img as *const _,
            x_blur: vec![0.; width * height],
            result: vec![0.; width * height],
            width,
            height,
            tasks,
        }
    }
}

unsafe impl Graph for Blur {}

#[bench]
fn blur(b: &mut Bencher) {
    let width = 10;
    let height = 10;
    let image = [032; width * height];
}
