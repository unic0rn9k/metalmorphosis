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
pub mod optimizer;
mod primitives;

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
    #[inline(always)]
    fn wake(self: Arc<Self>) {
        (*self).1.send(Signal::Wake(self.0)).unwrap()
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
    /// Return data from task to parent.
    ///
    /// # Safety
    /// The function is not type checked, so it's up to you to make sure the type of the read data matches the written data.
    #[inline(always)]
    pub unsafe fn write_output<O: MorphicIO>(&self, o: O) -> Result<(), T> {
        unsafe {
            if O::IS_COPY {
                std::ptr::write(self.output.fast_and_unsafe as *mut O, o)
            } else {
                *self.output.vec = bincode::serialize(&o)?
            }
        }
        Ok(())
    }

    pub async fn branch_copy<O: MorphicIO>(&self, token: T) -> Result<O, T> {
        unsafe {
            let optimizer_hint = (*self.optimizer).hint(&token);
            let parent = self.this_node;

            let mut buffer = O::buffer();
            let output = OutputSlice {
                fast_and_unsafe: &mut buffer as *mut O as *mut u8,
            };

            self.sender.send(Signal::Branch {
                token,
                parent,
                output,
                optimizer_hint,
            })?;

            halt_once().await;

            Ok(std::ptr::read(&buffer as *const O))
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
