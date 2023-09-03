<div align="center">
<img src="https://raw.githubusercontent.com/unic0rn9k/metalmorphosis/4th_refactor/logo.png" width="400"/>
</div>

This project lets you developer multi-threaded and distributed code, using async/await in Rust.
In Rust, async/await is designed to be implementation agnostic, meaning there is no asynchronous executor implemented in the language, instead it is up to library developers to develop executors, that specialize for a given task, like web-development, embedded or HPC.

This project aims to be a unified way, that allows to easily integrate schedulers, executors and profiling tools, at runtime or compile-time, into a codebase.

This is done by first building a task-graph of an async program, and then using that graph to supply the executor, and schedulers with information about the program. Schedulars can then mutate the graph, agregating profiling information at runtime (with builting or 3rd party methods), and then use this information later to inform decision on resource allocation and scheduling.

Currently, the project has a multi-threaded and distributed (built on MPI) executor. It also has a default multi-threaded scheduler, which is based on topological sorting, optimizing for both cache locality and maximum utilization of resources. 

The scheduler however has no information about the program, besides its graph representation, and so cannot be sure what tasks will require most resources and time.

Thus metalmorphosis allows to implement custom scheduling strategies, both to augment existing ones, or completely redefine how tasks are divided on a system or network.

## Note!
This branch contains a WIP rewrite of the original project, which is located at [the 4th refactor branch](https://github.com/unic0rn9k/metalmorphosis/tree/4th_refactor).
