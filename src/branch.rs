use crate::{buffer, error::Result, executor, BoxFuture, Edge, MorphicIO, TaskHandle};
use std::{future::Future, marker::PhantomData, sync::mpsc::Sender};
pub use OptHintSingle::*;

#[derive(Clone, Copy, Debug)]
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

pub enum Signal<'a> {
    //Wake(usize),
    Branch {
        // Should just send a TaskNode instead.
        program: BoxFuture<'a>,
        edge: Edge<'a>,
    },
}

unsafe impl<'a> std::marker::Send for Signal<'a> {}

pub struct Builder<'a, O: MorphicIO<'a>> {
    sender: Sender<Signal<'a>>,
    edge: Edge<'a>,
}

impl<'a, O: MorphicIO<'a>> Builder<'a, O> {
    fn hint(&mut self, hint: OptHintSingle) {
        self.opt_hint.add(hint);
    }
}

impl<'a, O: MorphicIO<'a>> Future for Builder<'a, O> {
    type Output = Result<'a, O>;

    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        let mut buffer = buffer::Source::uninit();
        let parent = self.this_node;

        let task_handle = TaskHandle::<'a, O> {
            sender: self.sender.clone(),
            output: buffer.alias(),
            opt_hint: self.opt_hint,
            phantom_data: PhantomData,
            // FIXME: This should not be initialized to 0
            this_node: 0,
        };

        self.sender.send(Signal::Branch {
            // Make a TypedTaskNode to pass as task_handle here.
            // This way we can use type checking inside of program and then convert to untyped task node, that can be passed to executor
            program: program(task_handle).extremely_unsafe_type_conversion(),
            edge: self.edge,
        })?;

        executor::halt_once().await;

        buffer.read()
    }
}
