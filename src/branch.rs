use crate::{
    buffer, error::Result, internal_utils::*, BoxFuture, Edge, MorphicIO, TaskHandle, Work,
};
use std::{future::Future, marker::PhantomData, sync::mpsc::Sender, task::Poll};
pub use OptHintSingle::*;

#[derive(Default, Clone, Copy, Debug)]
pub struct OptHint {
    always_serialize: bool,
    cache: bool,
    distribute: bool,
}

impl OptHint {
    fn add(&mut self, other: OptHintSingle) {
        match other {
            AlwaysSerialize => self.always_serialize = true,
            Cache => self.cache = true,
            Distribute => self.distribute = true,
        }
    }
}

pub enum OptHintSingle {
    AlwaysSerialize,
    Cache,
    Distribute,
}

// Maybe Signal, allong with some other stuff, should be moved to smth like edge.rs.
pub enum Signal<'a> {
    //Wake(usize),
    Branch {
        // Should just send a TaskNode instead.
        this_node: *mut usize,
        program: BoxFuture<'a>,
        edge: Edge<'a>,
    },
}

unsafe impl<'a> std::marker::Send for Signal<'a> {}

// Maybe Edge should be moved into builder, and then just keep a reference to a builder, instead of edges.
// Every node has a unique Builder anyway. This would mean TaskNode and TaskHandle would have to share a builder, tho.
pub struct Builder<'a, O: MorphicIO<'a>> {
    sender: Sender<Signal<'a>>,
    edge: Edge<'a>,
    buffer: buffer::Source<'a, O>,
    has_halted: bool,
    program: Work<'a>,
    this_node: usize,
    phantom_data: PhantomData<O>,
}

impl<'a, O: MorphicIO<'a>> Builder<'a, O> {
    pub fn hint(&mut self, hint: OptHintSingle) {
        self.edge.opt_hint.add(hint);
    }

    pub fn new<T: MorphicIO<'a>>(parent: &TaskHandle<'a, T>, program: Work<'a>) -> Self {
        // FIXME: Code bellow is very broken. And maybe there is a problem with safety in buffer::Alias
        //
        // code bellow produces UB:
        // fn bruh<'a>(_: &'a ()) -> buffer::Alias<'a>{
        //     let dude = buffer::Source::<'a, ()>::uninit();
        //     dude.alias()
        // }
        let mut tmp = Self {
            sender: parent.sender.clone(),
            edge: unsafe { uninit() },
            buffer: buffer::Source::uninit(),
            program,
            has_halted: false,
            this_node: 0,
            phantom_data: PhantomData,
        };
        tmp.edge = parent.new_edge(&tmp.buffer.alias());
        tmp
    }
}

impl<'a, O: MorphicIO<'a>> Future for Builder<'a, O> {
    type Output = Result<'a, O>;

    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        if self.has_halted {
            return Poll::Ready(self.buffer.read());
        }

        let parent = self.this_node;

        let task_handle = TaskHandle::<'a, O> {
            sender: self.sender.clone(),
            phantom_data: PhantomData,
            // FIXME: This should not be initialized to 0
            this_node: 0,
            edge: self.edge,
        };

        self.sender.send(Signal::Branch {
            // Make a TypedTaskNode to pass as task_handle here.
            // This way we can use type checking inside of program and then convert to untyped task node, that can be passed to executor
            program: self.program(task_handle).extremely_unsafe_type_conversion(),
            edge: self.edge,
            this_node: &mut self.this_node as *mut usize,
        })?;

        self.has_halted = true;

        return Poll::Pending;
    }
}
