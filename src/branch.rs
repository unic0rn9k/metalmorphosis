use crate::{buffer, error::Result, BoxFuture, Edge, MorphicIO, Task, TaskHandle};
use std::{future::Future, marker::PhantomData, task::Poll};
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

#[derive(Default, Clone, Copy, Debug)]
pub struct OptHint {
    pub serialize: bool,
    pub cache: bool,
    pub distribute: bool,
    pub run_on_root: bool,
    pub alloc: AllocationStrategy,
    pub reserved_branches: usize,
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

// Maybe Edge should be moved into builder, and then just keep a reference to a builder, instead of edges.
// Every node has a unique Builder anyway. This would mean TaskNode and TaskHandle would have to share a builder, tho.
pub struct Builder<'a, F: Task<'a, O>, O: MorphicIO<'a>> {
    handle: TaskHandle<'a, O>,
    buffer: buffer::Source<'a, O>,
    has_halted: bool,
    program: F,
}

impl<'a, F: Task<'a, O>, O: MorphicIO<'a>> Builder<'a, F, O> {
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

impl<'a, F: Task<'a, O>, O: MorphicIO<'a>> Future for Builder<'a, F, O> {
    type Output = Result<'a, O>;

    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        if self.has_halted {
            self.has_halted = false;
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
