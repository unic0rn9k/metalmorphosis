# Basic benchmarks
Under er et benchmark der viser overhead af executoren, sammenlignet med nogen hurtige operationer.
Benchmarked includere overhead af at initializere task-graphen.
```sh
test test::empty_vec    ... bench:           5 ns/iter (+/- 2)
test test::f_of_x_bench ... bench:     390,575 ns/iter (+/- 48,904)
test test::index        ... bench:           1 ns/iter (+/- 0)
test test::mull_add_1   ... bench:           6 ns/iter (+/- 2)
test test::mull_add_2   ... bench:          11 ns/iter (+/- 5)
test test::spawn_async  ... bench:           0 ns/iter (+/- 0)
```