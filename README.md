# metalmorphosis
Distributed async runtime in rust, with a focus on being able to build computation graphs (specifically auto-diff)


Benchmarks can be found at [benchmarks_og_bilag.md](benchmarks_og_bilag.md)

### TODO
- [X] Type checking
- [X] Buffers
- [X] impl Future for Symbol

- [X] handle for graph with type information about the node calling it.

- [X] Executor / schedular
    - Wakers? (wake me up inside)
- [X] multithreaded
    - join future that works with array of symbols

- [X] Distribute (OpenMPI?)
    - don't time awaits inside node
    - reusing output in node would confuse executor

- [ ] clean code (remove duplicate work)
- [ ] nicer API (ATLEAST for custom schedular)
- [ ] return Result everywhere
- [ ] if a child is pushed to pool, but all threads are occupied, prefer to poll it from the thread of the parent

- [X] priority que.
    - Let users set priority
    - increase priority of awaited children
    - internal events as tasks

- [ ] Benchmarks and tests
    - TRAVLT? just make a synthetic benchmark... `thread::sleep(Duration::from_millis(10))`

### Extra
- Resources can be used for different executor instances, at the same time, using PoolHandle and NetHandle.
- Anchored nodes (so that 0 isnt special. Then executor makes sure anchored nodes are done before kill)
- Mby do some box magic in Graph::output, so that MutPtr is not needed.
- Allocator reusablility for dynamic graphs
- Const graphs (lib.rs/phf)
- Time-complexity hints
- Static types for futures (allocate them on bump, and let node provide funktion pointer for polling)
- Graph serialization (need runtime typechecking for graph hot-realoading)
- Optional stack trace (basically already implemented this)
- Check for cycles when building graph
- Multiple backends for providing tasks (eg: shared object files, cranelift, fancy jit / hot-reloading)
- specialized optimisations based on graph structure, when initilizing (fx: combine multiple nodes, that only have a signle parent, into one node)
