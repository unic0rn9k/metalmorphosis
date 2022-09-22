#![allow(incomplete_features)]
#![feature(new_uninit)]
#![feature(get_mut_unchecked)]
#![feature(future_join)]
#![feature(let_else)]
#![feature(generic_const_exprs)]

#[cfg(test)]
mod tests;

use bincode;
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::intrinsics::transmute;
use std::pin::Pin;
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};

mod executor;
mod stupid_futures;
use stupid_futures::*;
mod error;
use error::*;
pub mod optimizer;

pub type BasicFuture = Box<dyn Future<Output = ()> + Unpin>;

pub union OutputSlice {
    vec: *mut Vec<u8>,
    fast_and_unsafe: *mut u8,
}

pub enum Signal<T: Program> {
    Wake(usize),
    Branch {
        token: T,
        parent: usize,
        output: OutputSlice,
        optimizer_hint: optimizer::HintFromOptimizer,
    },
}

unsafe impl<T: Program> std::marker::Send for Signal<T> {}

pub struct SignalWaker<T: Program>(usize, SyncSender<Signal<T>>);

impl<T: Program> Wake for SignalWaker<T> {
    fn wake(self: Arc<Self>) {
        self.as_ref().1.send(Signal::Wake(self.0)).unwrap()
    }
}

pub struct TaskNode<T: Program> {
    sender: SyncSender<Signal<T>>,
    output: OutputSlice,
    future: BasicFuture,
    parent: usize,
    this_node: usize,
    children: usize,
    optimizer: *const optimizer::Optimizer<T>,
    optimizer_hint: optimizer::HintFromOptimizer,
}

impl<T: Program> TaskNode<T> {
    pub unsafe fn write_output<O: MorphicIO>(&self, o: O) -> Result<(), T> {
        if O::IS_COPY {
            unsafe { std::ptr::write(self.output.fast_and_unsafe as *mut O, o) }
        } else {
            unsafe { *self.output.vec = bincode::serialize(&o)? }
        }
        Ok(())
    }

    pub async fn branch_copy<O: MorphicIO>(&self, token: T) -> Result<O, T> {
        unsafe {
            let optimizer_hint = (&*self.optimizer).hint(&token);
            let parent = self.this_node;

            let mut buffer = O::buffer();
            let output = OutputSlice {
                fast_and_unsafe: transmute(&mut buffer),
            };

            self.sender.send(Signal::Branch {
                token,
                parent,
                output,
                optimizer_hint,
            })?;

            halt_once().await;

            Ok(std::ptr::read(transmute(&buffer)))
        }
    }

    pub async fn branch_serialized<O: MorphicIO>(&self, token: T) -> Result<O, T> {
        let optimizer_hint = (unsafe { &*self.optimizer }).hint(&token);
        let parent = self.this_node;

        let mut buffer = Vec::with_capacity(0);
        let output = OutputSlice {
            vec: &mut buffer as &mut Vec<u8>,
        };

        self.sender.send(Signal::Branch {
            token,
            parent,
            output,
            optimizer_hint,
        })?;

        halt_once().await;

        Ok(bincode::deserialize(unsafe {
            std::mem::transmute(&mut buffer[..])
        })?)
    }

    pub async fn branch<O: MorphicIO>(&self, token: T) -> Result<O, T> {
        // Branch is called before the executor is allowed to run,
        // therefore we need a way to figure out if data should be distributed (and more) here.
        // This is what i atempted to do with the Optimizer struct, which might implemented optimally.
        println!(":     +__");
        println!(":     |  [{:?}]", token);
        println!(":     |  ");

        //let mut opt_hint = [0u8; 4];
        //self.sender.send(Signal::GetOptHint(
        //    self.this_node,
        //    token,
        //    opt_hint.as_mut_ptr(),
        //))?;
        //let opt_hint = optimization_hint().await;

        if O::IS_COPY {
            //&& opt_hint.is_local() {
            self.branch_copy(token).await
        } else {
            self.branch_serialized(token).await
        }
    }

    pub fn poll(&mut self) -> Poll<()> {
        Pin::new(&mut self.future).poll(&mut Context::from_waker(&Waker::from(Arc::new(
            SignalWaker(self.this_node, self.sender.clone()),
        ))))
    }
}

/// Trait that must be implemented for all valued passed between `TaskNode`s.
/// Will only be unsafe if IS_COPY is set to true.
pub unsafe trait MorphicIO: Serialize + Deserialize<'static> + Send + Sync {
    const IS_COPY: bool = false;
    //const SIZE: usize;
    //fn local_serialize(self, buffer: &mut [u8]);
    //fn local_deserialize(self, buffer: &[u8]);
    #[inline(always)]
    unsafe fn buffer() -> Self {
        unsafe { std::mem::MaybeUninit::uninit().assume_init() }
    }
}

pub trait Program: std::fmt::Debug + Send + Sync + Sized + 'static {
    type Future: Future<Output = ()> + Unpin + 'static;
    fn future(self, task_handle: Arc<TaskNode<Self>>) -> Self::Future;
}

#[inline(always)]
pub fn execute<T: Program>(program: T) -> Result<(), T> {
    executor::Executor::new().run(program)
}
