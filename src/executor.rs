use crate::{buffer, OptHint, Result, Signal, TaskHandle, TaskNode, Work};
use std::{
    future::Future,
    marker::PhantomData,
    sync::mpsc::{sync_channel, Receiver, SyncSender},
    task::Poll,
};

pub struct Executor<'a> {
    queue: Receiver<Signal<'a>>,
    self_sender: SyncSender<Signal<'a>>,
    task_graph: Vec<TaskNode<'a>>,
    leaf_nodes: Vec<usize>,
}

impl<'a> Executor<'a> {
    #[inline(always)]
    pub fn new() -> Self {
        let (self_sender, queue) = sync_channel(1000);
        Self {
            queue,
            self_sender,
            task_graph: vec![],
            leaf_nodes: vec![],
        }
    }

    pub fn run(&mut self, main: impl FnOnce(TaskHandle<'a, ()>) -> Work<'a>) -> Result<'a, ()> {
        self.branch(Signal::Branch {
            program: main(TaskHandle {
                sender: self.self_sender.clone(),
                output: buffer::null().alias(),
                this_node: 0,
                opt_hint: OptHint {
                    always_serialize: true,
                },
                phantom_data: PhantomData,
            })
            .extremely_unsafe_type_conversion(),
            parent: 0,
            output: buffer::null().alias(),
        });
        let mut n = 0;

        'polling: loop {
            if n == self.leaf_nodes.len() {
                // n = self.task_graph.len()-1; // ?
                n = 0;
                let mut branch = self.queue.try_recv();
                while branch.is_ok() {
                    self.branch(unsafe { branch.unwrap_unchecked() });
                    branch = self.queue.try_recv();
                }
            }

            let leaf = &mut self.task_graph[self.leaf_nodes[n]];
            if leaf.poll().is_ready() {
                if self.leaf_nodes[n] == leaf.parent {
                    // self.task_graph.clear();
                    return Ok(());
                } else {
                    let parent = leaf.parent;
                    // This would change the relative indecies of all nodes :(
                    // self.task_graph.remove(leaf.this_node);
                    self.task_graph[parent].children -= 1;
                    if self.task_graph[parent].children == 0 {
                        self.leaf_nodes[n] = parent;
                    } else {
                        self.leaf_nodes.remove(n);
                    }
                    continue 'polling;
                }
            }

            n += 1;
        }
    }

    pub fn branch(&mut self, branch: Signal<'a>) {
        let Signal::Branch {
            parent,
            output,
            program,
        } = branch else{todo!()};

        match self.task_graph.get_mut(parent) {
            None => {}
            Some(parent) => parent.children += 1,
        }

        // TODO: This does not have to be this slow...
        for n in 0..self.leaf_nodes.len() {
            if self.leaf_nodes[n] == parent {
                self.leaf_nodes.remove(n);
                break;
            }
        }

        self.leaf_nodes.push(self.task_graph.len());
        self.task_graph.push(TaskNode {
            sender: self.self_sender.clone(),
            output,
            future: program,
            this_node: self.task_graph.len(),
            parent,
            children: 0,
            opt_hint: OptHint {
                // Do we need to send data over network?
                always_serialize: false,
            },
        });
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
