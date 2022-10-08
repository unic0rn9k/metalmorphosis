//! # metalmorposis
//! Distributed async runtime in rust, with a focus on being able to build computation graphs (specifically auto-diff).
//!
//! examples can be found in examples directory.
//!
//! # Weird place to have a todo list...
//! - Maybe rename MorphicIO back to Distributed or distributable.
//! - Think of a better name than "program". It's more like a node, or smth.
//! - examples/math.rs (AutoDiff)
//! - src/network.rs (distribute that bitch)
//! - I removed wakers again
//! - What was the point of all these lifetimes again?

#![feature(new_uninit, future_join, type_alias_impl_trait)]

use serde::{Deserialize, Serialize};
use std::{
    future::Future,
    marker::PhantomData,
    pin::Pin,
    sync::{mpsc::SyncSender, Arc},
    task::{Context, Poll, Wake, Waker},
};

pub mod autodiff;
pub mod error;
pub mod executor;
use error::*;
mod buffer;
//mod network;
mod primitives;

pub type BoxFuture<'a> = Pin<Box<dyn Future<Output = ()> + Unpin + 'a>>;
//pub type Program<'a> = impl FnOnce(&TaskNode<'a>) -> Work<'a>;

#[derive(Clone, Copy, Debug)]
pub struct OptHint {
    pub always_serialize: bool,
}

pub enum Signal<'a> {
    //Wake(usize),
    Branch {
        // Should just send a TaskNode instead.
        program: BoxFuture<'a>,
        parent: usize,
        output: buffer::Alias<'a>,
    },
}

unsafe impl<'a> std::marker::Send for Signal<'a> {}

/*
pub struct SignalWaker<T: Program>(usize, SyncSender<Signal<T>>);

impl<T: Program> Wake for SignalWaker<T> {
    #[inline(always)]
    fn wake(self: Arc<Self>) {
        (*self).1.send(Signal::Wake(self.0)).unwrap()
    }
}
*/

pub struct TaskNode<'a> {
    sender: SyncSender<Signal<'a>>,
    output: buffer::Alias<'a>,
    future: BoxFuture<'a>,
    this_node: usize,
    parent: usize,
    children: usize,
    opt_hint: OptHint,
}

struct NullWaker;

impl Wake for NullWaker {
    fn wake(self: Arc<Self>) {
        todo!()
    }
}

pub struct TaskHandle<'a, T: MorphicIO<'a>> {
    // Maybe this should also include a reference to its coresponding TaskNode?
    sender: SyncSender<Signal<'a>>,
    output: buffer::Alias<'a>,
    this_node: usize,
    opt_hint: OptHint,
    phantom_data: PhantomData<T>,
}

impl<'a> TaskNode<'a> {
    pub fn poll(&mut self) -> Poll<()> {
        Pin::new(&mut self.future).poll(&mut Context::from_waker(&Waker::from(Arc::new(NullWaker))))
    }
}

impl<'a, T: MorphicIO<'a>> TaskHandle<'a, T> {
    /// Return data from task to parent.
    #[inline(always)]
    pub fn output(self, o: T) -> Result<'a, ()> {
        unsafe {
            let buffer = self.output.attach_type();
            if T::IS_COPY && !self.opt_hint.always_serialize {
                // Raw data (just move it)
                buffer.set_data_format::<'r'>()
            } else {
                // Serialized data
                buffer.set_data_format::<'s'>()
            }
            Ok(buffer.write(o)?)
        }
    }

    // Should be able to take an iterater of programs,
    // that also describe edges,
    // that way we can just append a whole existing graph at once.
    //
    // Maybe branches should be unsafe?
    pub async fn branch<O: MorphicIO<'a>>(
        &self,
        program: impl FnOnce(TaskHandle<'a, O>) -> Work<'a>,
    ) -> Result<O> {
        // This is actually also unsafe, if the child doesn't write any data, and the parent tries to read it.
        let mut buffer = buffer::Source::uninit();
        let parent = self.this_node;

        let task_handle = TaskHandle::<'a, O> {
            sender: self.sender.clone(),
            output: buffer.alias(),
            opt_hint: OptHint {
                // Do we need to send data over network?. Idk if we can know this here.
                always_serialize: !O::IS_COPY,
            },
            phantom_data: PhantomData,
            // FIXME: This should not be initialized to 0
            this_node: 0,
        };

        self.sender.send(Signal::Branch {
            // Make a TypedTaskNode to pass as task_handle here.
            // This way we can use type checking inside of program and then convert to untyped task node, that can be passed to executor
            program: program(task_handle).extremely_unsafe_type_conversion(),
            parent,
            output: buffer.alias(),
        })?;

        executor::halt_once().await;

        buffer.read()
    }

    // This is gonna create problems in the future (haha thats a type)
    // if we dont make sure theres some information about which device its from,
    // and some safety checks.
    #[inline(always)]
    pub fn node_id(&self) -> usize {
        self.this_node
    }

    pub async fn attach_tree<O: MorphicIO<'a>>(tree: Vec<TaskNode<'a>>) -> Result<'a, O> {
        // Husk at `parent` skal shiftes, så det passer med de nye relative positioner i `task_graph`
        todo!()
    }

    /*
    pub async fn orphan(&self, this_node: usize) -> TaskNode<'a> {
        Self {
            sender: self.sender.clone(),
            output: todo!(),
            future: todo!(),
            parent: self.this_node,
            this_node,
            children: 0,
            opt_hint: todo!(),
        }
    }
    */
}

/// Trait that must be implemented for all valued passed between `TaskNode`s.
///
/// # Safety
/// Make sure `IS_COPY` is only true for types that implement copy.
pub unsafe trait MorphicIO<'a>: 'a + Serialize + Deserialize<'a> + Send + Sync {
    // Think this while copy thing might be redundant, now that buffer is more safe.
    // Still defs are some safety concerns to concider.
    const IS_COPY: bool = false;
    /// DON'T OVERWRITE THIS FUNCTION.
    /// Returns a buffer that can fit Self, for use internally.
    ///
    /// # Safety
    /// As long as `IS_COPY` is set correctly, theres no problem.
    #[inline(always)]
    unsafe fn buffer() -> Self {
        if Self::IS_COPY {
            unsafe { std::mem::MaybeUninit::uninit().assume_init() }
        } else {
            panic!("Tried to create buffer for non-copy data")
        }
    }
}

pub struct Work<'a>(Pin<Box<dyn Future<Output = ()> + 'a>>);

impl<'a> Work<'a> {
    fn extremely_unsafe_type_conversion(self) -> BoxFuture<'a> {
        unsafe { std::mem::transmute(self.0) }
    }
}

#[inline(always)]
pub fn execute<'a>(program: impl FnOnce(TaskHandle<'a, ()>) -> Work<'a>) {
    executor::Executor::new().run(program).unwrap()
}

#[inline(always)]
pub fn work<'a>(f: impl Future<Output = ()> + 'a) -> Work<'a> {
    Work(Box::pin(f))
}
