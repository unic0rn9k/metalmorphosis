<div align="center">
<h1> metalmorphosis </h1>
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

The easiest way to handle data return might be with distributed side-effects tho.
Just make buffer::Alias serializable and contain a machine-id.
Then when you want to write to it, it might just send the pointer and data to the machine with the id,
which will then write the data.
This will of course likely only work if the data is in the serialized format.

it should be possible to do *Prefetching* of distributed pointer values.
Meaning if we know that 'this device' is gonna read from 'other device',
and other device already has the value ready.
then it would make sense to schedule a read from other device,
even tho this device doesn't need the value yet.
