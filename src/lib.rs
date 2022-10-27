//! # metalmorphosis
//! Distributed async runtime in rust, with a focus on being able to build computation graphs (specifically auto-diff).
//!
//! examples can be found in examples directory.
//!
//! # Weird place to have a todo list...
//! - Maybe rename MorphicIO back to Distributed or distributable.
//! - examples/math.rs (AutoDiff)
//! - src/network.rs (distribute that bitch)
//! - I removed wakers again
//! - Mixed static and dynamic graphs. (Describe location of static node based on displacement from dynamic parent node)
//! - Node caching
//!
//! # Project timeline
//! 0. Auto-diff graph (linear algebra mby)
//! 1. multi-threaded (Static graphs, node caching)
//! 2. distributed (mio and buffer/executor changes)
//! 3. Route optimization (also when should caching occur? maybe just tell explicitly when :/)

#![feature(new_uninit, future_join, type_alias_impl_trait)]

use branch::OptHint;
use buffer::{RAW, SERIALIZED};
use error::*;
use internal_utils::*;
use serde::{Deserialize, Serialize};
use std::{
    future::Future,
    marker::PhantomData,
    pin::Pin,
    sync::{mpsc::Sender, Arc},
    task::{Context, Poll, Wake, Waker},
};

pub mod autodiff;
mod buffer;
pub mod error;
pub mod executor;
//mod static_graph;
//mod network;
mod branch;
mod internal_utils;
mod primitives;

pub type BoxFuture<'a> = Pin<Box<dyn Future<Output = ()> + Unpin + 'a>>;

// Should be able to take a generic input, in addition to the handle.
pub trait Task<'a, O: MorphicIO<'a>>: FnOnce(TaskHandle<'a, O>) -> Work<'a> {}

/*
pub struct SignalWaker<T: Program>(usize, Sender<Signal<T>>);

impl<T: Program> Wake for SignalWaker<T> {

    fn wake(self: Arc<Self>) {
        (*self).1.send(Signal::Wake(self.0)).unwrap()
    }
}
*/

pub struct Edge<'a> {
    this_node: usize,
    parent: usize,
    output: buffer::Alias<'a>,
    opt_hint: OptHint,
}

pub struct TaskNode<'a> {
    future: BoxFuture<'a>,
    children: usize,
    edge: Edge<'a>,
}

struct NullWaker;

impl Wake for NullWaker {
    fn wake(self: Arc<Self>) {
        todo!()
    }
}

pub struct TaskHandle<'a, T: MorphicIO<'a>> {
    // Maybe this should also include a reference to its coresponding TaskNode?
    sender: Sender<branch::Signal<'a>>,
    edge: Edge<'a>,
    phantom_data: PhantomData<T>,
}

impl<'a> TaskNode<'a> {
    pub fn poll(&mut self) -> Poll<()> {
        Pin::new(&mut self.future).poll(&mut Context::from_waker(&Waker::from(Arc::new(NullWaker))))
    }
}

impl<'a, T: MorphicIO<'a>> TaskHandle<'a, T> {
    /// Return data from task to parent.

    pub fn output(self, o: T) -> Result<'a, ()> {
        unsafe {
            let buffer = self.edge.output.attach_type();
            if T::IS_COPY && !self.edge.opt_hint.serialize {
                // Raw data (just move it)
                buffer.set_data_format::<RAW>()
            } else {
                // Serialized data
                buffer.set_data_format::<SERIALIZED>()
            }
            Ok(buffer.write(o)?)
        }
    }

    // Should be able to take an iterater of programs,
    // that also describe edges,
    // that way we can just append a whole existing graph at once.
    //
    // Maybe branches should be unsafe?
    pub async fn branch<F: Task<'a, O>, O: MorphicIO<'a>>(
        &self,
        program: F,
    ) -> branch::Builder<'a, F, O> {
        branch::Builder::new(self, program)
    }

    pub fn new_edge(&self) -> Edge<'a> {
        Edge {
            output: buffer::null(),
            // NOTE: self.this_node is used here, but is not properly initialized (branch.rs).
            parent: self.edge.this_node,
            opt_hint: OptHint::default(),
            // This may cause trouble in the future...
            this_node: 0,
        }
    }

    // This is gonna create problems in the future (haha thats a type)
    // if we dont make sure theres some information about which device its from,
    // and some safety checks.
    pub fn node_id(&self) -> usize {
        self.edge.this_node
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

    unsafe fn buffer() -> Self {
        if Self::IS_COPY {
            unsafe { uninit() }
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

pub fn execute<'a>(program: impl FnOnce(TaskHandle<'a, ()>) -> Work<'a>) {
    executor::Executor::new().run(program).unwrap()
}

pub fn work<'a>(f: impl Future<Output = ()> + 'a) -> Work<'a> {
    Work(Box::pin(f))
}
