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

#![feature(new_uninit, future_join)]

use serde::{Deserialize, Serialize};
use std::{
    future::Future,
    pin::Pin,
    sync::{mpsc::SyncSender, Arc},
    task::{Context, Poll, Wake, Waker},
};

pub mod autodiff;
pub mod error;
pub mod executor;
use error::*;
mod buffer;
mod network;
mod primitives;

pub type BoxFuture<'a> = Pin<Box<dyn Future<Output = ()> + Unpin + 'a>>;

#[derive(Clone, Copy, Debug)]
pub struct OptHint {
    pub always_serialize: bool,
}

pub enum Signal<'a, T: Program<'a>> {
    //Wake(usize),
    Branch {
        program: T,
        parent: usize,
        output: buffer::Alias<'a>,
    },
}

unsafe impl<'a, T: Program<'a>> std::marker::Send for Signal<'a, T> {}

/*
pub struct SignalWaker<T: Program>(usize, SyncSender<Signal<T>>);

impl<T: Program> Wake for SignalWaker<T> {
    #[inline(always)]
    fn wake(self: Arc<Self>) {
        (*self).1.send(Signal::Wake(self.0)).unwrap()
    }
}
*/

pub struct TaskNode<'a, T: Program<'a>> {
    sender: SyncSender<Signal<'a, T>>,
    output: buffer::Alias<'a>,
    future: BoxFuture<'a>,
    parent: usize,
    this_node: usize,
    children: usize,
    opt_hint: OptHint,
}

struct NullWaker;

impl Wake for NullWaker {
    fn wake(self: Arc<Self>) {
        unreachable!()
    }
}

impl<'a, T: Program<'a>> TaskNode<'a, T> {
    pub fn poll(&mut self) -> Poll<()> {
        Pin::new(&mut self.future).poll(&mut Context::from_waker(&Waker::from(Arc::new(NullWaker))))
    }

    /// Return data from task to parent.
    ///
    /// # Safety
    /// The function is not type checked, so it's up to you to make sure the type of the read data matches the written data.
    #[inline(always)]
    pub unsafe fn output<O: MorphicIO<'a>>(&'a self, o: O) -> Result<'a, (), T> {
        let buffer = self.output.attach_type();
        if O::IS_COPY && !self.opt_hint.always_serialize {
            // Raw data (just move it)
            buffer.set_data_format::<'r'>()
        } else {
            // Serialized data
            buffer.set_data_format::<'s'>()
        }
        Ok(buffer.write(o)?)
    }

    // Should be able to take an iterater of programs,
    // that also describe edges,
    // that way we can just append a whole existing graph at once.
    #[inline(always)]
    pub async fn branch<O: MorphicIO<'a>>(&'a self, program: impl Into<T>) -> Result<O, T> {
        // This is actually also unsafe, if the child doesn't write any data, and the parent tries to read it.
        let mut buffer = buffer::Source::uninit();
        let output = buffer.alias();
        let parent = self.this_node;

        self.sender.send(Signal::Branch {
            program: program.into(),
            parent,
            output,
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

pub struct Work<'a>(Box<dyn Future<Output = ()> + 'a>);

impl<'a> Work<'a> {
    fn extremely_unsafe_type_conversion(self) -> BoxFuture<'a> {
        unsafe { std::mem::transmute(self.0) }
    }
}

pub trait Program<'a>: std::fmt::Debug + Send + Sync + Sized {
    fn future<T: Program<'a> + From<Self>>(self, task_handle: &'a TaskNode<'a, T>) -> Work<'a>;
}

#[inline(always)]
pub fn execute<'a, T: Program<'a> + 'a>(program: T) -> Result<'a, (), T> {
    executor::Executor::new().run(program)
}

#[inline(always)]
pub fn work<'a>(f: impl Future<Output = ()> + 'a) -> Work<'a> {
    Work(Box::new(f))
}
