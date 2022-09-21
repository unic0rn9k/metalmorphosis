#![feature(new_uninit)]
#![feature(get_mut_unchecked)]
#![feature(future_join)]
#![feature(let_else)]

#[cfg(test)]
mod tests;

use bincode;
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

pub type BasicFuture = Box<dyn Future<Output = ()> + Unpin>;
pub union OutputSlice {
    vec: *mut Vec<u8>,
    fast_and_unsafe: *mut u8,
}

pub enum Signal<T: Program> {
    Wake,
    Branch {
        token: T,
        parent: usize,
        output: OutputSlice,
        optimizer_hint: optimizer::HintFromOptimizer,
    },
}

pub struct TaskNode<T: Program> {
    sender: SyncSender<Signal<T>>,
    output: OutputSlice,
    future: BasicFuture,
    parent: usize,
    this_node: usize,
    children: usize,
    token: T,
    optimizer: *const optimizer::Optimizer<T>,
    optimizer_hint: optimizer::HintFromOptimizer,
}

struct NilWaker;

impl Wake for NilWaker {
    fn wake(self: Arc<Self>) {
        panic!("`std::task::context` context isn't supported rn...");
    }
}

impl<T: Program> TaskNode<T> {
    pub fn write_output<O: MorphicIO>(&self, o: O) -> Result<(), T> {
        if self.optimizer_hint.fast_and_unsafe_serialization {
            unsafe { std::ptr::write(self.output.fast_and_unsafe as *mut O, o) }
        } else {
            unsafe { *self.output.vec = bincode::serialize(&o)? }
        }
        Ok(())
    }

    pub async fn branch<O: MorphicIO>(&self, token: T) -> Result<O, T> {
        println!("      +__");
        println!("      |  [{:?}]", token);
        println!("      |  ");

        let optimizer_hint = (unsafe { &*self.optimizer }).hint(token);
        let parent = self.this_node;
        let (mut buffer, output);

        if optimizer_hint.fast_and_unsafe_serialization {
            let o_size = std::mem::size_of::<O>();
            buffer = Vec::with_capacity(o_size);
            output = OutputSlice {
                fast_and_unsafe: buffer.as_mut_ptr(),
            };
        } else {
            buffer = Vec::with_capacity(0);
            output = OutputSlice {
                vec: &mut buffer as &mut Vec<u8>,
            };
        };

        self.sender.send(Signal::Branch {
            token,
            parent,
            output,
            optimizer_hint,
        })?;

        halt_once().await;

        if optimizer_hint.fast_and_unsafe_serialization {
            unsafe { Ok(std::ptr::read(buffer.as_ptr() as *const O)) }
        } else {
            Ok(bincode::deserialize(unsafe {
                std::mem::transmute(&mut buffer[..])
            })?)
        }
    }

    pub fn poll(&mut self) -> Poll<()> {
        Pin::new(&mut self.future).poll(&mut Context::from_waker(&Waker::from(Arc::new(NilWaker))))
    }
}

pub trait MorphicIO: Serialize + Deserialize<'static> {
    //const SIZE: usize;
    //fn local_serialize(self, buffer: &mut [u8]);
    //fn local_deserialize(self, buffer: &[u8]);
}

pub trait Program: Copy + std::fmt::Debug {
    type Future: Future<Output = ()> + Unpin + 'static;
    fn future(self, task_handle: Arc<TaskNode<Self>>) -> Self::Future;
}

#[inline(always)]
pub fn execute<T: Program>(program: T) -> Result<(), T> {
    executor::Executor::new().run(program)
}
