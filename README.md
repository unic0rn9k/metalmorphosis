<div align="center">
<img src="https://raw.githubusercontent.com/unic0rn9k/metalmorphosis/4th_refactor/logo.png" width="300"/>
</div>

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

- [ ] return Result everywhere
- [X] handle for graph with type information about the node calling it.

- [ ] Executor / schedular
    - Wakers? (wake me up inside)
- [ ] multithreaded
    - thread pool
    - join future that works with array of symbols
- [ ] Benchmark two-stage blur

- [ ] Distribute (OpenMPI?)
    - don't time awaits inside node
    - reusing output in node would confuse executor
- [ ] Benchmark distributed

### Extra
- Allocator reusablility for dynamic graphs
- Const graphs (lib.rs/phf)
- Time-complexity hints
- Static types for futures
- Graph serialization (need runtime typechecking for graph hot-realoading)
- Optional stack trace (basically already implemented this)
- Check for cycles when building graph
- Multiple backends for providing tasks (eg: shared object files, cranelift, fancy jit / hot-reloading)
