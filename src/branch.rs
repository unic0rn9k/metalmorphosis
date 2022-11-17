use crate::{buffer, error::Result, BoxFuture, Edge, MorphicIO, Task, TaskHandle};
use std::{future::Future, marker::PhantomData, pin::Pin, task::Poll};
pub use OptHintSingle::*;

#[derive(Debug, Clone, Copy, Default)]
pub enum AllocationStrategy {
    #[default]
    Dynamic,
    DynamicWithReserved(usize),
    Static(usize),
}

impl Into<OptHintSingle> for AllocationStrategy {
    fn into(self) -> OptHintSingle {
        AllocStrategy(self)
    }
}

// Need two hint types.
// - One to hint about children, from parent
// - And one to hint about self from self
#[derive(Default, Clone, Copy, Debug)]
pub struct OptHint {
    pub serialize: bool,
    pub cache: bool,
    pub distribute: bool,
    pub run_on_root: bool,
    pub alloc: AllocationStrategy,
    pub reserved_branches: usize,
    pub keep_local: bool,
}

impl OptHint {
    fn add(&mut self, other: OptHintSingle) {
        match other {
            AlwaysSerialize => self.serialize = true,
            Cache => self.cache = true,
            Distribute => self.distribute = true,
            AllocStrategy(a) => self.alloc = a,
            // TODO: these should not do the same thing.
            HardReserveBranches(n) => self.reserved_branches = n,
            SoftReserveBranches(n) => self.reserved_branches = n,
            RunOnRoot => self.run_on_root = true,
            RunOnAll => todo!(),
            KeepLocal => self.keep_local = true,
        }
    }
}

pub enum OptHintSingle {
    /// Just serialized the output
    AlwaysSerialize,
    /// Safe buffer for storing temporary values, and other stuff for later use.
    /// This is usefull if you await the same branch multiple times.
    /// for example like you would do in a for loop.
    /// Might me useless in the future.
    Cache,
    // Agresively distribute task
    Distribute,
    // This task should always be run on the root device.
    // This is for example useful for comunicating data to a human.
    // If youre connected to a cluster with your laptop, and you want some plots, you can make sure theire saved to the right device.
    RunOnRoot,
    // Run this on all devices. Usefull for synchronizing data.
    RunOnAll,
    // Do you wanna use a ringbuffer? now you can
    AllocStrategy(AllocationStrategy),
    // Reserve branches for static-graph allocation.
    HardReserveBranches(usize),
    // Do you have some idea of how many branches a node will spawn? Tell the executor, then.
    SoftReserveBranches(usize),
    // Just await the future directly. This way you can avoid a heap allocation,
    // if you don't need to distribute or multithread a task
    KeepLocal,
}

// Maybe Signal, along with some other stuff, should be moved to smth like edge.rs.
pub enum Signal<'a> {
    //Wake(usize),
    Branch {
        // Should just send a TaskNode instead.
        program: BoxFuture<'a>,

        // TODO: Executor needs to atleast set `this_node` correctly.
        // Props also a good idea to do some stuff with opt_hint.
        edge: *mut Edge<'a>,
    },
}

unsafe impl<'a> std::marker::Send for Signal<'a> {}

pub struct Builder<'a, F: Task<'a, O>, O: MorphicIO<'a>> {
    handle: TaskHandle<'a, O>,
    // This should possible be a reference to some preallocated buffer space.
    buffer: buffer::Source<'a, O>,
    has_halted: bool,
    program: F,
}

impl<'a, F: Task<'a, O>, O: MorphicIO<'a>> Builder<'a, F, O> {
    // FIXME: Det giver ikke helt mening... Hvordan skal man give hints om børn fra parent?
    // Det giver kun mening for en specifik type hints. Så som DontDistribute.
    // Tror children bliver nød til at kunne lave special setup stuff,
    // hvis der ikke skal være en seperat static graph trait, og så med meget slow dynamic graphs
    pub fn hint<I: IntoIterator<Item = OptHintSingle>>(&mut self, hint: I) {
        for hint in hint {
            self.handle.edge.opt_hint.add(hint);
        }
    }

    pub fn new<T: MorphicIO<'a>>(parent: &TaskHandle<'a, T>, program: F) -> Self {
        // FIXME: Maybe there is a problem with safety in buffer::Alias
        //
        // code bellow produces UB:
        // fn bruh<'a>(_: &'a ()) -> buffer::Alias<'a>{
        //     let dude = buffer::Source::<'a, ()>::uninit();
        //     dude.alias()
        // }
        Self {
            handle: TaskHandle {
                sender: parent.sender.clone(),
                phantom_data: PhantomData,
                edge: parent.new_edge(),
                preallocated_children: 0,
            },
            buffer: buffer::Source::uninit(),
            program,
            has_halted: false,
        }
    }
}

// If a branch should allocate it's own output,
// then it should be able to spawn multiple nodes (and props take a generic argument of type Allocator).
//
//
// Otherwise the parent task handle needs to provide it with an allocator to use.
impl<'a, F: Task<'a, O>, O: MorphicIO<'a>> Future for Builder<'a, F, O> {
    type Output = Result<'a, O>;

    // # Recursive executor
    // The executor should be contain in the branch.
    // There are two graphs, one for data and one for polling.
    // This should figure out if we should:
    // - continuously poll same task until its ready.
    // - poll parent when child is pending.
    // - assign to thread pool, and let child poll parent when its done.
    // - poll some network stuff mby.
    // - Do something completely different?

    // Synchronous reusability can always be executed thread localy
    // a -poll-> b
    // b -retu-> a
    // b -poll-> a
    // a use value from b
    // a -poll-> b
    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        if self.handle.edge.opt_hint.keep_local {
            self.handle.edge.output = self.buffer.alias();
            let mut future = (self.program)(self.handle).extremely_unsafe_type_conversion();
            let mut status = Pin::new(&mut future).poll(cx);
            while status == Poll::Pending {
                status = Pin::new(&mut future).poll(cx);
            }
            return Poll::Ready(self.buffer.read());
        }

        if self.has_halted {
            self.has_halted = false;
            // Check if buffer has been written to.
            // If this is a vertically reusable node,
            // then it needs to make sure it has been updated since it was last read.
            //
            // Then we can have another object that spawns a collection of nodes and functions as an iterator of Builders
            // Here it can just manually set the has_halted flag in the builder, to avoid it re-spawning the nodes.
            return Poll::Ready(self.buffer.read());
        }

        self.handle.edge.output = self.buffer.alias();

        // Maybe this should be moved up to `new`
        self.handle.sender.push(
            // Make a TypedTaskNode to pass as task_handle here.
            // This way we can use type checking inside of program and then convert to untyped task node,
            // that can be passed to executor
            (self.program)(self.handle),
            &mut self.handle.edge,
        );

        self.has_halted = true;

        return Poll::Pending;
    }
}

// # NEED TO DO WAKERS
// {
//   spawn bunch of task (self.waker)
//   ret Pending
// }
//
// child{
//   do work and write res
//   parent.wake
// }
//
// Also, this reminds me way to much of a mpsc::Sender, figure out how that works under the hood.
