use crate::{buffer, OptHint, Program, Result, Signal, TaskNode};
use std::{
    future::Future,
    sync::mpsc::{sync_channel, Receiver, SyncSender},
    task::Poll,
};

pub struct Executor<'a, T: Program<'a>> {
    queue: Receiver<Signal<'a, T>>,
    self_sender: SyncSender<Signal<'a, T>>,
    task_graph: Vec<TaskNode<'a, T>>,
}

impl<'a, T: Program<'a> + 'a> Executor<'a, T> {
    #[inline(always)]
    pub fn new() -> Self {
        let (self_sender, queue) = sync_channel(1000);
        Self {
            queue,
            self_sender,
            task_graph: vec![],
        }
    }

    pub fn run(&mut self, main: T) -> Result<'a, (), T> {
        #[allow(const_item_mutation)]
        self.branch(Signal::Branch {
            program: main,
            parent: 0,
            output: buffer::NULL.alias(),
        });
        let mut n = 0;

        'polling: loop {
            if n == self.task_graph.len() {
                n = 0;
                let mut branch = self.queue.try_recv();
                while branch.is_ok() {
                    self.branch(unsafe { branch.unwrap_unchecked() });
                    branch = self.queue.try_recv();
                }
            }

            if self.task_graph[n].children != 0 {
                n += 1;
                continue 'polling;
            }

            if self.task_graph[n].poll().is_ready() {
                if self.task_graph[n].this_node == self.task_graph[n].parent {
                    self.task_graph.clear();
                    return Ok(());
                } else {
                    let parent = self.task_graph[n].parent;
                    self.task_graph.remove(n);
                    n = parent;
                    self.task_graph[n].children -= 1;
                    continue 'polling;
                }
            }

            n += 1;
        }
    }

    pub fn branch(&mut self, branch: Signal<'a, T>) {
        let Signal::Branch {
            parent,
            output,
            program,
        } = branch else{todo!()};

        match self.task_graph.get_mut(parent) {
            None => {}
            Some(parent) => parent.children += 1,
        }

        let node = TaskNode {
            sender: self.self_sender.clone(),
            output,
            future: Box::pin(halt_once()),
            parent,
            this_node: self.task_graph.len(),
            children: 0,
            opt_hint: OptHint {
                // Do we need to send data over network?
                always_serialize: false,
            },
        };

        self.task_graph.push(node);
        let last = self.task_graph.len() - 1;

        let task_handle = &self.task_graph[last] as *const TaskNode<'a, T>;

        self.task_graph[last].future = program
            .future::<T>(unsafe { &*task_handle })
            .extremely_unsafe_type_conversion();
    }
}

/// A future that returns Pending once, and then Ready. This let's the executor do its thing.
pub struct HaltOnceWaker(bool);

impl Future for HaltOnceWaker {
    type Output = ();

    #[inline(always)]
    fn poll(
        mut self: std::pin::Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        if self.0 {
            Poll::Ready(())
        } else {
            self.as_mut().0 = true;
            Poll::Pending
        }
    }
}

pub fn halt_once() -> HaltOnceWaker {
    HaltOnceWaker(false)
}
