extern crate test;
use std::ops::Index;

use serde::{Deserialize, Serialize};
use test::{black_box, Bencher};

use crate::{builder::GraphBuilder, builder::Task, Symbol};

fn sample(dim: &[usize; 2]) -> Vec<f32> {
    let mut sample = vec![0f32; dim[0] * dim[1]];
    for x in 0..dim[0] {
        let k = (dim[1] as f32 / 2.) - 2.;
        let y = (x as f32 * 0.1).sin() * k + k + 2.;
        sample[x + y as usize * dim[0]] = 1.;
    }
    sample
}

fn table(img: &impl Index<usize, Output = f32>, dim: &[usize]) {
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
    fn blur_trans(img: &[f32], output: &mut [f32], dim: &[usize; 2]) {
        let img = |x: isize, y: isize| {
            if x < 0 || x >= dim[0] as isize || y < 0 || y >= dim[1] as isize {
                0f32
            } else {
                black_box(img[black_box(x as usize + y as usize * dim[0])])
            }
        };

        let [x, y] = dim;
        for x in 0..*x as isize {
            black_box(x);
            for y in 0..*y as isize {
                let p = (img(x + 1, y) + img(x - 1, y) + img(x, y)) / 3.;
                // output[x as usize + y as usize * dim[0]] = p; // non-transposed output
                *black_box(&mut output[y as usize + x as usize * dim[1]]) = black_box(p);
                // transposed output
            }
        }
    }

    let input = black_box(sample(&DIM));
    let mut horizontal = black_box(vec![0f32; DIM[0] * DIM[1]]);

    b.iter(|| {
        let mut output = black_box(vec![0f32; DIM[0] * DIM[1]]);
        let trans = [DIM[1], DIM[0]];
        blur_trans(&input, &mut horizontal, &DIM);
        blur_trans(&horizontal, &mut output, &trans);

        black_box(&output[..]);
    });

    //table(&input, &DIM);
    //table(&output, &DIM);
}

struct Const<T>(*const T);
impl<T: Sync + Serialize + Deserialize<'static> + 'static> Task for Const<T> {
    type InitOutput = Symbol<T>;
    type Output = T;

    fn init(self, graph: &mut GraphBuilder<Self>) -> Self::InitOutput {
        graph.mutate_this_node(|node| unsafe {
            let b = &mut *node.output.get();
            b.data = self.0 as *mut ();
            b.drop = |_| {};
        });
        graph.this_node()
    }
}

#[derive(Clone, Copy)]
struct Matrix(*const f32, [usize; 2]);
unsafe impl Send for Matrix {}
impl Index<[usize; 2]> for Matrix {
    type Output = f32;

    fn index(&self, index: [usize; 2]) -> &Self::Output {
        let [x, y] = index;
        assert!(x < self.1[0] && y < self.1[1]);
        unsafe { &*self.0.add(x + y * self.1[0]) }
    }
}

struct MorphicBlur<'a>(&'a Vec<f32>, &'a mut Vec<f32>, &'a mut Vec<f32>, [usize; 2]);
struct MorphicBlurStage {
    input: Symbol<Vec<f32>>,
    dim: [usize; 2],
    bound: [[usize; 2]; 2],
}

impl Task for MorphicBlurStage {
    type InitOutput = Symbol<Vec<f32>>;
    type Output = Vec<f32>;

    fn init(self, graph: &mut GraphBuilder<Self>) -> Self::InitOutput {
        let source = graph.lock_symbol(self.input);

        graph.task(Box::new(move |graph, node| {
            let source = source.clone().own(graph);
            Box::pin(async move {
                let m = unsafe { Matrix((*source.await.0).as_ptr(), self.dim) };
                for y in self.bound[0][1]..self.bound[1][1] {
                    for x in self.bound[0][0]..self.bound[1][0] {
                        let p = (m[[x + 1, y]] + m[[x - 1, y]] + m[[x, y]]) / 3.;
                        unsafe {
                            (*node.output::<Vec<f32>>())[x * m.1[1] + y] = p;
                        }
                    }
                }
            })
        }));
        graph.this_node()
    }
}

impl<'a> Task for MorphicBlur<'a> {
    type InitOutput = ();
    type Output = ();

    fn init(self, graph: &mut GraphBuilder<Self>) -> Self::InitOutput {
        let dim = self.3;

        let input = graph.spawn(Const(self.0), None);

        let chunks = 2;
        let chunk = (dim[1] - 1) / chunks;
        let mut output = vec![];

        for n in 1..chunks + 1 {
            let stage1 = graph.spawn(
                MorphicBlurStage {
                    input: input.clone(),
                    dim,
                    bound: [[1, 1], [dim[0] - 1, n * chunk]],
                },
                Some(self.1),
            );

            let out = graph.spawn(
                MorphicBlurStage {
                    input: stage1.clone(),
                    dim: [dim[1], dim[0]],
                    bound: [[1, 1], [n * chunk, dim[0] - 1]],
                },
                Some(self.2),
            );
            output.push(graph.lock_symbol(out));
        }

        graph.task(Box::new(move |graph, _node| {
            let mut output: Vec<_> = output.iter().map(|s| s.clone().own(graph)).collect();
            Box::pin(async move {
                //println!("=== main polled ===");
                for out in output.drain(..) {
                    //println!("another blur awaited");
                    black_box(out.await);
                    //unsafe {
                    //    table(&*out.await.0, &dim);
                    //}
                }

                //println!("=== main done ===");
            })
        }))
    }
}

fn morphic_blur<'a>(
    input: &'a Vec<f32>,
    stage1: &'a mut Vec<f32>,
    output: &'a mut Vec<f32>,
    dim: &[usize; 2],
) -> MorphicBlur<'a> {
    MorphicBlur(input, stage1, output, *dim)
}

#[bench]
fn morphic(b: &mut Bencher) {
    let padded_dim = [DIM[0] + 2, DIM[1] + 2];
    let input = sample(&padded_dim);
    let mut stage1 = vec![0f32; padded_dim[0] * padded_dim[1]];
    let mut output = vec![0f32; padded_dim[0] * padded_dim[1]];

    let builder = GraphBuilder::main(morphic_blur(&input, &mut stage1, &mut output, &padded_dim));
    let graph = builder.build();
    let (net_events, mut net) = graph.init_net();
    b.iter(|| {
        graph.spin_down();
        graph.realize(net_events.clone());
        net.run();
    });
    graph.kill(net.kill());
}

const DIM: [usize; 2] = [8000, 80];
