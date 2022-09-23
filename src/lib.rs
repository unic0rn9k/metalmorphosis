#![allow(incomplete_features)]
#![feature(new_uninit)]
#![feature(get_mut_unchecked)]
#![feature(future_join)]
#![feature(let_else)]
#![feature(generic_const_exprs)]

use serde::{Deserialize, Serialize};
use std::future::Future;
use std::pin::Pin;
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};

mod executor;
mod stupid_futures;
use stupid_futures::*;
mod error;
use error::*;
pub mod buffer;
//pub mod optimizer;
mod primitives;

pub type BasicFuture = Box<dyn Future<Output = ()> + Unpin>;

#[derive(Clone, Copy, Debug)]
pub struct OptHint {
    pub always_serialize: bool,
}

pub enum Signal<T: Program> {
    Wake(usize),
    GetOptHint(),
    Branch {
        token: T,
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
    future: BasicFuture,
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
    pub unsafe fn write_output<O: MorphicIO>(&self, o: O) -> Result<(), T> {
        let buffer = self.output.attach_type();
        if O::IS_COPY && !self.opt_hint.always_serialize {
            buffer.set_data_format('r')
        } else {
            buffer.set_data_format('s')
        }
        Ok(buffer.write(o)?)
    }

    #[inline(always)]
    pub async fn branch<O: MorphicIO>(&self, token: T) -> Result<O, T> {
        // Branch is called before the executor is allowed to run,
        // therefore we need a way to figure out if data should be distributed (and more) here.
        // This is what i atempted to do with the Optimizer struct, which might implemented optimally.
        #[cfg(profile = "debug")]
        {
            println!(":     +__");
            println!(":     |  [{:?}]", token);
            println!(":     |  ");
        }

        //let mut opt_hint = [0u8; 4];
        //self.sender.send(Signal::GetOptHint(
        //    self.this_node,
        //    token,
        //    opt_hint.as_mut_ptr(),
        //))?;
        //halt_once().await;

        let mut buffer = buffer::Source::uninit();
        let output = buffer.alias();
        let parent = self.this_node;

        self.sender.send(Signal::Branch {
            token,
            parent,
            output,
        })?;

        halt_once().await;

        unsafe { buffer.read() }
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
    //const SIZE: usize;
    //fn local_serialize(self, buffer: &mut [u8]);
    //fn local_deserialize(self, buffer: &[u8]);

    /// DON'T OVERWRITE THIS FUNCTION.
    /// Returns a buffer that can fit Self, for use internally.
    ///
    /// # Safety
    /// As long as `IS_COPY` is set correctly, theres no problem.
    #[inline(always)]
    unsafe fn buffer() -> Self {
        unsafe { std::mem::MaybeUninit::uninit().assume_init() }
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
