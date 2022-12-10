<div align="center">
<h1> metalmorphosis </h1>
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
    - Props shouldn't return a raw-ptr tho 😬

- [ ] return Result everywhere
- [X] handle for graph with type information about the node calling it.

- [ ] Executor / schedular
    - Wakers? (wake me up inside)
- [ ] Benchmark two-stage blur
- [ ] Distribute (OpenMPI?)
    - don't time awaits inside node
    - reusing output in node would confuse executor
- [ ] Benchmark distributed

### Extra
- Allocator reusablility for dynamic graphs
- const graphs
- time-complexity hints
- static types for futures
