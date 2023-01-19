extern crate test;
use test::{black_box, Bencher};

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
                let p = (img(x + 1, y) + img(x - 1, y) + img(x, y)) / 3.;
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
