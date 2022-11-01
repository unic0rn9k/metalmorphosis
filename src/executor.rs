use dashmap::DashMap;

use crate::{
    branch::Signal, buffer, OptHint, Result, TaskGraph, TaskHandle, TaskNode, Work, ROOT_EDGE,
};
use std::{
    future::Future,
    marker::PhantomData,
    sync::{
        atomic::AtomicUsize,
        mpsc::{channel, Receiver, Sender},
        Arc,
    },
    task::Poll,
};

pub struct Executor<'a>
where
    Self: 'a,
{
    // DashMap
    task_graph: Arc<TaskGraph<'a>>,
    // TODO: leaf_nodes do a lot of pointles heap allocations
    leaf_nodes: Vec<usize>,
}

impl<'a> Executor<'a> {
    pub fn new() -> Self {
        Self {
            task_graph: TaskGraph::new(),
            leaf_nodes: vec![],
        }
    }

    pub fn run(&mut self, main: impl FnOnce(TaskHandle<'a, ()>) -> Work<'a>) -> Result<'a, ()> {
        let root_handle = TaskHandle {
            sender: self.task_graph.clone(),
            edge: ROOT_EDGE,
            phantom_data: PhantomData,
        };
        self.task_graph.push(main(root_handle), &mut ROOT_EDGE);

        let mut n = 0;

        'polling: loop {
            //if n == self.leaf_nodes.len() {
            //    // n = self.task_graph.len()-1; // ?
            //    n = 0;
            //    let mut branch = self.queue.try_recv();
            //    while branch.is_ok() {
            //        self.branch(unsafe { branch.unwrap_unchecked() });
            //        branch = self.queue.try_recv();
            //    }
            //}

            let leaf = &mut self.task_graph[self.leaf_nodes[n]];
            if leaf.poll().is_ready() {
                if self.leaf_nodes[n] == leaf.edge.parent {
                    // self.task_graph.clear();
                    return Ok(());
                } else {
                    let parent = leaf.edge.parent;
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

    /*
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
            output,
            future: program,
            parent,
            children: 0,
            /*opt_hint: OptHint {
                // Do we need to send data over network?
                always_serialize: false,
            },*/
        });
    }
    */
}

/// A future that returns Pending once, and then Ready. This let's the executor do its thing.
pub struct HaltOnceWaker(bool);

impl Future for HaltOnceWaker {
    type Output = ();

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
