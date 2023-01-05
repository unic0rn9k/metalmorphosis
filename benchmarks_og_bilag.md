# Basic micro-benchmarks
Under er et benchmark der viser overhead af executoren, sammenlignet med nogen hurtige operationer.
Benchmarked includere overhead af at initializere task-graphen.
```sh
$ cargo bench
test test::atomic_load ... bench:           1 ns/iter (+/- 0)
test test::empty_vec   ... bench:           5 ns/iter (+/- 2)
test test::f_of_x_old  ... bench:     420,575 ns/iter (+/- 48,904)
test test::f_of_x      ... bench:     172,820 ns/iter (+/- 49,560)
test test::index       ... bench:           1 ns/iter (+/- 0)
test test::mull_add_1  ... bench:           7 ns/iter (+/- 3)
test test::mull_add_2  ... bench:          13 ns/iter (+/- 6)
test test::send_recv   ... bench:         392 ns/iter (+/- 156)
test test::spawn_async ... bench:           0 ns/iter (+/- 0)
```

efter at have valgt bedere Atomic Orderings blev `f_of_x` significant hurtigere.
```sh
test test::f_of_x       ... bench:     257,757 ns/iter (+/- 44,328)
```

# Pure MPI micro-benchmark
Under er koden og resultaterne af et simpelt benchmark der poviser MPI's performance når man komunikkere mellem threads på den same maskine.

```rust
extern crate mpi;
use std::{mem::transmute, time::Instant};

use mpi::traits::*;

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let size = world.size();
    let rank = world.rank();

    if size != 2 {
        panic!("Size of MPI_COMM_WORLD must be 2, but is {}!", size);
    }

    match rank {
        0 => {
            let time: [u8; 16] = unsafe { transmute(Instant::now()) };
            world.process_at_rank(rank + 1).send(&time);

            let time: [u8; 16] = unsafe { transmute(Instant::now()) };
            world.process_at_rank(rank + 1).send(&time);
        }
        1 => {
            let (time, st) = world.any_process().receive_vec::<u8>();
            //assert_eq!(time.len(), 16);
            let time: &Instant = unsafe { transmute(&time[0]) };
            println!("time in nano seconds: {}", time.elapsed().as_nanos());
            println!("{st:?}");

            let (time, st) = world.any_process().receive_vec::<u8>();
            //assert_eq!(time.len(), 16);
            let time: &Instant = unsafe { transmute(&time[0]) };
            println!("time in nano seconds: {}", time.elapsed().as_nanos());
            println!("{st:?}");
        }
        _ => unreachable!(),
    }
}
```

```sh
$ mpiexec --hostfile hosts cargo r --release
     ...lots of lines omitted...
    Finished release [optimized] target(s) in 0.18s
     Running `target/release/mpi_sutff`
    Finished release [optimized] target(s) in 0.21s
     Running `target/release/mpi_sutff`
time in nano seconds: 5404064
Status { source_rank: 0, tag: 0 }
time in nano seconds: 102028
Status { source_rank: 0, tag: 0 }
```