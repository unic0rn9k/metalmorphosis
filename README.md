<div align="center">
<img src="https://raw.githubusercontent.com/unic0rn9k/metalmorphosis/4th_refactor/logo.png" width="400"/>
</div>

Benchmarks can be found at [benchmarks_og_bilag.md](benchmarks_og_bilag.md)

### Definitions
- Symbol: a type used to refer to a node,
  that can be bound to another node, returning a future to the output of a node.
  (it lets you specify edges in the computation graph)

- Dealocks will be caused by:
`graph.attach_edge(Self::edge(graph));`,
`graph.spawn(F(Self::edge(graph)));`.

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

- [X] priority que.
    - Let users set priority
    - increase priority of awaited children
    - internal events as tasks

- [ ] Benchmarks and tests
    - TRAVLT? just make a synthetic benchmark... `thread::sleep(Duration::from_millis(10))`

### Extra
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
