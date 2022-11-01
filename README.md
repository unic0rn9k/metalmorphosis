<div align="center">
## metalmorphosis
</div>

Distributed async runtime in rust, with a focus on being able to build computation graphs (specifically auto-diff).

examples can be found in examples directory.

### Weird place to have a todo list...
- Maybe rename MorphicIO back to Distributed or distributable.
- examples/math.rs (AutoDiff)
- src/network.rs (distribute that bitch)
- I removed wakers again
- Mixed static and dynamic graphs. (Describe location of static node based on displacement from dynamic parent node)
- Node caching

### Project timeline
0. Auto-diff graph (linear algebra mby)
1. multi-threaded (Static graphs, node caching)
2. distributed (mio and buffer/executor changes)
3. Route optimization (also when should caching occur? maybe just tell explicitly when :/)

### Distributed pointers
Function side-effects are very inefficient on a distributed system,
as there is no way to directly mutate data on another device.
