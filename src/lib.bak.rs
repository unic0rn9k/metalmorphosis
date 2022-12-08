//! <div align="center">
//! <h1> metalmorphosis </h1>
//! </div>
//!
//! Distributed async runtime in rust, with a focus on being able to build computation graphs (specifically auto-diff).
//!
//! examples can be found in examples directory.
//!
//! ## Weird place to have a todo list...
//! - Maybe rename MorphicIO back to Distributed or distributable.
//! - examples/math.rs (AutoDiff)
//! - src/network.rs (distribute that bitch)
//! - I removed wakers again
//! - Mixed static and dynamic graphs. (Describe location of static node based on displacement from dynamic parent node)
//! - Node caching
//!
//! ## Project timeline
//! 0. Auto-diff graph (linear algebra mby)
//! 1. multi-threaded (Static graphs, node caching)
//! 2. distributed (mio and buffer/executor changes)
//! 3. Route optimization (also when should caching occur? maybe just tell explicitly when :/)
//!
//! ## Distributed pointers
//! Function side-effects are very inefficient on a distributed system,
//! as there is no way to directly mutate data on another device.
//!
//! The easiest way to handle data return might be with distributed side-effects tho.
//! Just make buffer::Alias serializable and contain a machine-id.
//! Then when you want to write to it, it might just send the pointer and data to the machine with the id,
//! which will then write the data.
//! This will of course likely only work if the data is in the serialized format.
//!
//! it should be possible to do *Prefetching* of distributed pointer values.
//! Meaning if we know that 'this device' is gonna read from 'other device',
//! and other device already has the value ready.
//! then it would make sense to schedule a read from other device,
//! even tho this device doesn't need the value yet.
//!
//! # Implement me
//! https://play.rust-lang.org/?version=nightly&mode=debug&edition=2021&code=use%20std%3A%3Aops%3A%3A*%3B%0A%0Astruct%20TaskHandle%3B%0A%0Atrait%20TaskHandleProvider%3C%27a%3E%7B%0A%20%20%20%20fn%20handle(%26%27a%20mut%20self)%20-%3E%20%26%27a%20mut%20TaskHandle%3B%0A%7D%0A%0Aimpl%3C%27a%3E%20TaskHandleProvider%3C%27a%3E%20for%20TaskHandle%7B%0A%20%20%20%20%23%5Binline(always)%5D%0A%20%20%20%20fn%20handle(%26%27a%20mut%20self)%20-%3E%20%26%27a%20mut%20Self%7B%0A%20%20%20%20%20%20%20%20self%0A%20%20%20%20%7D%0A%7D%0A%0Aimpl%3C%27a%2C%20I%3A%20Iterator%3CItem%3D%26%27a%20mut%20TaskHandle%3E%3E%20TaskHandleProvider%3C%27a%3E%20for%20I%7B%0A%20%20%20%20%23%5Binline(always)%5D%0A%20%20%20%20fn%20handle(%26%27a%20mut%20self)%20-%3E%20%26%27a%20mut%20TaskHandle%7B%0A%20%20%20%20%20%20%20%20self.next().unwrap()%0A%20%20%20%20%7D%0A%7D%0A%0Astruct%20IntoTask%3B%0A%0Aimpl%20IntoTask%7B%0A%20%20%20%20%2F%2F%20Maybe%20this%20should%20return%20an%20Iterator%3Cimpl%20Fn%3E%0A%20%20%20%20%2F%2F%20https%3A%2F%2Fdocs.rs%2Ffutures%2Flatest%2Ffutures%2Fstream%2Ftrait.Stream.html%0A%20%20%20%20fn%20task%3C%27a%2C%20H%3A%20TaskHandleProvider%3C%27a%3E%3E(%26self%2C%20handle%3A%20H)-%3E%20impl%20Fn()%7B%0A%20%20%20%20%20%20%20%20move%20%7C%7Cprintln!(%22ok%22)%0A%20%20%20%20%7D%0A%7D%0A%0Afn%20spawn%3CF%3A%20Fn()%2C%20T%3A%20Fn(TaskHandle)-%3EF%3E(task%3A%20T)%7B%0A%20%20%20%20task(TaskHandle)()%0A%7D%0A%0Afn%20main()%7B%0A%20%20%20%20let%20a%20%3D%20IntoTask%3B%0A%20%20%20%20for%20n%20in%200..10%7B%0A%20%20%20%20%20%20%20%20spawn(%7Chandle%7Ca.task(handle))%0A%20%20%20%20%7D%0A%7D%0A%0A%0A%2F%2F%20Synchronous%20reusability%20is%20the%20same%20as%3A%0A%2F%2F%20fn(task)%7B%0A%2F%2F%20%20%20parent.send(handle.branch(task).hint(KeepLocal))%3B%0A%2F%2F%20%20%20parent.poll%0A%2F%2F%20%7D%0A%2F%2F%0A%2F%2F%20It%20would%20ever%20make%20sence%20to%20do%20this%20over%20a%20network%2C%0A%2F%2F%20as%20the%20executor%20would%20never%20be%20able%20to%20re-distribute%20the%20task%20to%20a%20new%20device%0A%2F%2F%20whihch%20itself%20would%20never%20have%20enough%20overhead%20(compared%20to%20the%20networking%20of%20sending%20data)%0A%2F%2F%20to%20actually%20justify%20not%20doing%20it.%0A%2F%2F%0A%2F%2F%20Thus%20it%20would%20always%20be%20better%20to%20just%20branch%20in%20a%20loop.%0A%2F%2F%20%0A%2F%2F%20Given%20that%20it%20can%20be%20avoided%20to%20run%20setup%20multiple%20times.

#![feature(
    new_uninit,
    future_join,
    type_alias_impl_trait,
    const_type_name,
    trait_alias
)]

use branch::OptHint;
use buffer::{RAW, SERIALIZED};
use error::*;
use internal_utils::*;
use primitives::Array;
use serde::{Deserialize, Serialize};
use std::{
    any::type_name,
    future::Future,
    marker::PhantomData,
    pin::Pin,
    sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc::Sender,
        Arc,
    },
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

// Functions in rust are stupid. Cant do multiple dispatch the cool way eg: fn bruh(a: f32); fn bruh(a: &str)
// ```rust
// trait Task<'a, O: MorphicIO<'a>>: MorphicIO<'a>{
//      fn hint() -> Hint{
//          How many childre we have?
//          How many bytes should be preallocated?
//          Then combine this with some hints from the parent. And send that bitch to the executor.
//      }
//      fn work(&mut self, handle: TaskHandle<'a, O>) -> Work<'a>;
// }
// ```
//pub trait Task<'a, O: MorphicIO<'a>>: FnOnce(TaskHandle<'a, O>) -> Work<'a> {
//    const NAME: &'static str = type_name::<Self>();
//}

// This might have to be a trait (or Fn), instead of a function pointer,
// to guarantee that it is possible to build an efficient ad-graph on top.
pub trait Task<'a, O: MorphicIO<'a>> = Fn(TaskHandle<'a, O>) -> Work<'a>;

/*
pub struct SignalWaker<T: Program>(usize, Sender<Signal<T>>);

impl<T: Program> Wake for SignalWaker<T> {

    fn wake(self: Arc<Self>) {
        (*self).1.send(Signal::Wake(self.0)).unwrap()
    }
}
*/

#[derive(Clone, Copy)]
pub struct Edge<'a> {
    this_node: usize,
    parent: usize,
    output: buffer::Alias<'a>,
    opt_hint: OptHint,
}

const ROOT_EDGE: Edge<'static> = Edge {
    this_node: 0,
    parent: 0,
    output: null_alias!(),
    opt_hint: OptHint::default(),
};

// TODO: There should be some concept of a reusable node, that can do the same thing multiple times.
// All nodes should be reusable, but they should be able to implement the looping themselves,
// if it can be done more efficiently like that.
// Otherwise it should just reconstrukt the node for every iteration.
pub struct TaskNode<'a> {
    // TODO: Maybe a name field, for debugging. Should just be `type_name::<F: Task>()`
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
    // needs to be able to both poll parent directly,
    // Send wake signal for parent to executor,
    // And send a wake signal to parent thread directly,
    // based on OptHints and context.
    // parent_waker: GenericWakerType,

    // Maybe this should also include a reference to its corresponding TaskNode?
    sender: Sender<branch::Signal<'a>>,
    edge: Edge<'a>,
    phantom_data: PhantomData<T>,
    // TODO: Make sure you dont insert a task into a spot that already contains a running task.
    preallocated_children: usize,
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
            // TODO: Notify parent that data has been written.
        }
    }

    // Should be able to take an iterator of programs,
    // that also describe edges,
    // that way we can just append a whole existing graph at once.
    //
    // Should props take a SourceProvider as argument,
    // and have an option to reuse SourceProvider from parent for static graphs.
    pub async fn branch<F: Task<'a, O>, O: MorphicIO<'a>>(
        &self,
        program: F,
    ) -> branch::Builder<'a, F, O> {
        branch::Builder::new(self, program)
    }

    pub fn new_edge(&self) -> Edge<'a> {
        Edge {
            output: null_alias!(),
            // NOTE: self.this_node is used here, but is not properly initialized (branch.rs).
            parent: self.edge.this_node,
            opt_hint: OptHint::default(),
            // This may cause trouble in the future...
            this_node: 0,
        }
    }

    // This is gonna create problems in the future (haha thats a type)
    // if we don't make sure theres some information about which device its from,
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
