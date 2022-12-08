## Definitions
- Symbol: a name used to refer to a node. A future to the output of a node
  This has a concept of scope,
  meaning a given symbol might not refer to the same value in all nodes.

- Dealocks will be caused by:
`graph.attach_edge(Self::edge(graph));`
`graph.spawn(F(Self::edge(graph)));`

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
- dynamic graphs: a node that might spawn an unknown amount of sub nodes
- Node array: a struct that points to a node and lets you index into the nodes children
