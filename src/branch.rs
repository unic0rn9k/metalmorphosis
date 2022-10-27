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
        AllocStrat(self)
    }
}

#[derive(Default, Clone, Copy, Debug)]
pub struct OptHint {
    pub serialize: bool,
    pub cache: bool,
    pub distribute: bool,
    pub alloc: AllocationStrategy,
    pub branches: usize,
}

impl OptHint {
    fn add(&mut self, other: OptHintSingle) {
        match other {
            AlwaysSerialize => self.serialize = true,
            Cache => self.cache = true,
            Distribute => self.distribute = true,
            AllocStrat(a) => self.alloc = a,
            Branches(n) => self.branches = n,
        }
    }
}

pub enum OptHintSingle {
    AlwaysSerialize,
    Cache,
    Distribute,
    AllocStrat(AllocationStrategy),
    Branches(usize),
}

// Maybe Signal, allong with some other stuff, should be moved to smth like edge.rs.
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
        self.handle.sender.send(Signal::Branch {
            // Make a TypedTaskNode to pass as task_handle here.
            // This way we can use type checking inside of program and then convert to untyped task node, that can be passed to executor
            program: (self.program)(self.handle).extremely_unsafe_type_conversion(),
            edge: &mut self.handle.edge as *mut Edge<'a>,
        })?;

        self.has_halted = true;

        return Poll::Pending;
    }
}
