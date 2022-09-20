#![feature(new_uninit)]
#![feature(get_mut_unchecked)]
#![feature(future_join)]

#[cfg(test)]
mod tests;

use rmp_serde as rmps;
use rmps::Serializer;
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

pub type BasicFuture = Box<dyn Future<Output = ()> + Unpin>;
pub type OutputSlice = (*mut u8, usize);

pub struct TaskNode<T: Program> {
    sender: SyncSender<(T, usize, OutputSlice)>,
    output: OutputSlice,
    future: BasicFuture,
    parent: usize,
    this_node: usize,
    token: T,
    children: usize,
    // Should maybe also contain an "optimizer hint"
}

struct NilWaker;

impl Wake for NilWaker {
    fn wake(self: Arc<Self>) {
        panic!("`std::task::context` context isn't supported rn...");
    }
}

impl<T: Program> TaskNode<T> {
    pub fn write_output<O: MorphicIO>(&self, o: O) -> Result<(), T> {
        let buffer = unsafe { std::slice::from_raw_parts_mut(self.output.0, self.output.1) };
        Ok(o.serialize(&mut Serializer::new(buffer))?)
    }

    pub async fn branch<O: MorphicIO>(&self, token: T) -> Result<O, T> {
        println!("      +__");
        println!("      |  [{:?}]", token);
        println!("      |  ");
        let o_size = std::mem::size_of::<O>();
        let mut buffer = vec![0u8; o_size];
        self.sender
            .send((token, self.this_node, (&mut buffer[0] as *mut u8, o_size)))?;
        halt_once().await;
        Ok(rmps::from_slice(unsafe {
            std::mem::transmute(&mut buffer[..])
        })?)
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
    fn future(&self, task_handle: Arc<TaskNode<Self>>) -> Self::Future;
}

#[inline(always)]
pub fn execute<T: Program>(program: T) -> Result<(), T> {
    executor::Executor::new().run(program)
}
