//! # Weird place to have a todo list...
//! - Maybe rename MorphicIO back to Distributed or distributable.
//! - Think of a better name than "program". It's more like a node, or smth.
//! - examples/math.rs (AutoDiff)
//! - src/network.rs (distribute that bitch)

#![feature(new_uninit, future_join)]

use serde::{Deserialize, Serialize};
use std::{
    future::Future,
    pin::Pin,
    sync::{mpsc::SyncSender, Arc},
    task::{Context, Poll, Wake, Waker},
};

pub mod error;
mod executor;
use error::*;
mod buffer;
mod network;
mod primitives;

pub type BoxFuture = Box<dyn Future<Output = ()> + Unpin>;

#[derive(Clone, Copy, Debug)]
pub struct OptHint {
    pub always_serialize: bool,
}

pub enum Signal<T: Program> {
    Wake(usize),
    Branch {
        program: T,
        parent: usize,
        output: buffer::Alias,
    },
}

unsafe impl<T: Program> std::marker::Send for Signal<T> {}

pub struct SignalWaker<T: Program>(usize, SyncSender<Signal<T>>);

impl<T: Program> Wake for SignalWaker<T> {
    #[inline(always)]
    fn wake(self: Arc<Self>) {
        (*self).1.send(Signal::Wake(self.0)).unwrap()
    }
}

pub struct TaskNode<T: Program> {
    sender: SyncSender<Signal<T>>,
    output: buffer::Alias,
    future: BoxFuture,
    parent: usize,
    this_node: usize,
    children: usize,
    opt_hint: OptHint,
}

impl<T: Program> TaskNode<T> {
    /// Return data from task to parent.
    ///
    /// # Safety
    /// The function is not type checked, so it's up to you to make sure the type of the read data matches the written data.
    #[inline(always)]
    pub unsafe fn output<O: MorphicIO>(&self, o: O) -> Result<(), T> {
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

    #[inline(always)]
    pub async fn branch<O: MorphicIO>(&self, program: T) -> Result<O, T> {
        let mut buffer = buffer::Source::uninit();
        let output = buffer.alias();
        let parent = self.this_node;

        self.sender.send(Signal::Branch {
            program,
            parent,
            output,
        })?;

        executor::halt_once().await;

        buffer.read()
    }

    pub fn poll(&mut self) -> Poll<()> {
        Pin::new(&mut self.future).poll(&mut Context::from_waker(&Waker::from(Arc::new(
            SignalWaker(self.this_node, self.sender.clone()),
        ))))
    }
}

/// Trait that must be implemented for all valued passed between `TaskNode`s.
///
/// # Safety
/// Make sure `IS_COPY` is only true for types that implement copy.
pub unsafe trait MorphicIO: Serialize + Deserialize<'static> + Send + Sync {
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

pub trait Program: std::fmt::Debug + Send + Sync + Sized + 'static {
    type Future: Future<Output = ()> + Unpin + 'static;
    fn future(self, task_handle: &'static TaskNode<Self>) -> Self::Future;
}

#[inline(always)]
pub fn execute<T: Program>(program: T) -> Result<(), T> {
    executor::Executor::new().run(program)
}
