<div align="center">
<h1> metalmorphosis </h1>
</div>

## Definitions
- Symbol: a type used to refer to a node,
  that can be bound to another node, returning a future to the output of a node.
  (it lets you specify edges in the computation graph)

- Dealocks will be caused by:
`graph.attach_edge(Self::edge(graph));`,
`graph.spawn(F(Self::edge(graph)));`.

## TODO
- [ ] type checking
- [ ] awaiting nodes (buffer stuff, etc)
- [ ] runnable (executor)
- [ ] Benchmark two-stage blur
- [ ] Distribute (OpenMPI?)
    - don't time awaits inside node
    - reusing output in node would confuse executor
- [ ] Benchmark distributed

## Extra
- Allocator reusablility for dynamic graphs
