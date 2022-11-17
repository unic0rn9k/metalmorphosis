//! a way to store tasks
//! poll tasks from different threads
//! push tasks from different threads
//! await tasks from different threads
//! partial await (for sequential reusability)
//!
//! cross-device polling
//! same-derive polling
//!
//! # Scheduler needs to handle 3 cases
//! ## All resources or most devices (including self), are free
//! - keep local
//! - distribute
//! - wait
//!
//! ## This device is occupied
//! - distribute
//! - wait
//!
//! ## All resources are bussy
//! - wait

enum ResourceAvailability {
    OnlySelf,
    OthersAndSelf,
    Others,
    None,
}

enum DistributionStrategy {
    KeepLocal,
    Distribute,
    Wait,
}

// BashMap playground:
// https://play.rust-lang.org/?version=stable&mode=debug&edition=2021&code=use%20std%3A%3A%7B%0A%20%20%20%20collections%3A%3AHashMap%2C%0A%20%20%20%20ops%3A%3A%7BDeref%2C%20DerefMut%7D%2C%0A%20%20%20%20sync%3A%3A%7B%0A%20%20%20%20%20%20%20%20atomic%3A%3A%7BAtomicUsize%2C%20Ordering%7D%2C%0A%20%20%20%20%20%20%20%20Arc%2C%0A%20%20%20%20%7D%2C%0A%20%20%20%20thread%2C%0A%7D%3B%0A%0Ause%20spin%3A%3ARwLock%3B%0A%0Astruct%20BashMap%20%7B%0A%20%20%20%20src%3A%20Arc%3CRwLock%3CHashMap%3Cusize%2C%20usize%3E%3E%3E%2C%0A%20%20%20%20len%3A%20AtomicUsize%2C%0A%7D%0A%0Aimpl%20BashMap%20%7B%0A%20%20%20%20fn%20new()%20-%3E%20Self%20%7B%0A%20%20%20%20%20%20%20%20Self%20%7B%0A%20%20%20%20%20%20%20%20%20%20%20%20src%3A%20Arc%3A%3Anew(RwLock%3A%3Anew(HashMap%3A%3Anew()))%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20len%3A%20AtomicUsize%3A%3Anew(0)%2C%0A%20%20%20%20%20%20%20%20%7D%0A%20%20%20%20%7D%0A%0A%20%20%20%20fn%20reserve(%26mut%20self%2C%20capacity%3A%20usize)%20%7B%0A%20%20%20%20%20%20%20%20self.src.write().reserve(capacity)%0A%20%20%20%20%7D%0A%0A%20%20%20%20fn%20insert(%26mut%20self%2C%20k%3A%20usize%2C%20v%3A%20usize)%20%7B%0A%20%20%20%20%20%20%20%20let%20delta%20%3D%0A%20%20%20%20%20%20%20%20%20%20%20%20self.read().capacity()%20as%20isize%20-%20self.len.fetch_add(1%2C%20Ordering%3A%3ASeqCst)%20as%20isize%3B%0A%20%20%20%20%20%20%20%20if%20delta%20%3C%201%20%7B%0A%20%20%20%20%20%20%20%20%20%20%20%20print!(%22Reserving%20delta%20for%20%7Bk%7D...%20%22)%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20self.reserve((1%20-%20delta)%20as%20usize)%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20println!(%22%20ok%22)%3B%0A%20%20%20%20%20%20%20%20%7D%0A%0A%20%20%20%20%20%20%20%20let%20reader%20%3D%20self.src.read()%3B%0A%20%20%20%20%20%20%20%20unsafe%20%7B%0A%20%20%20%20%20%20%20%20%20%20%20%20%23%5Ballow(mutable_transmutes)%5D%0A%20%20%20%20%20%20%20%20%20%20%20%20let%20leak%3A%20%26mut%20HashMap%3Cusize%2C%20usize%3E%20%3D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20std%3A%3Amem%3A%3Atransmute(spin%3A%3ARwLockReadGuard%3A%3Aleak(self.src.read()))%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%2F%2Flet%20mut%20leak%20%3D%20self.src.write()%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20leak.insert(k%2C%20v)%3B%0A%20%20%20%20%20%20%20%20%7D%0A%20%20%20%20%20%20%20%20drop(reader)%0A%20%20%20%20%7D%0A%0A%20%20%20%20fn%20handle(%26mut%20self)%20-%3E%20BashRef%20%7B%0A%20%20%20%20%20%20%20%20BashRef(self%20as%20*mut%20Self)%0A%20%20%20%20%7D%0A%7D%0A%0Aimpl%20Deref%20for%20BashMap%20%7B%0A%20%20%20%20type%20Target%20%3D%20RwLock%3CHashMap%3Cusize%2C%20usize%3E%3E%3B%0A%0A%20%20%20%20fn%20deref(%26self)%20-%3E%20%26Self%3A%3ATarget%20%7B%0A%20%20%20%20%20%20%20%20%26self.src%0A%20%20%20%20%7D%0A%7D%0A%0A%23%5Bderive(Copy%2C%20Clone)%5D%0Astruct%20BashRef(*mut%20BashMap)%3B%0A%0Aimpl%20Deref%20for%20BashRef%20%7B%0A%20%20%20%20type%20Target%20%3D%20BashMap%3B%0A%0A%20%20%20%20fn%20deref(%26self)%20-%3E%20%26Self%3A%3ATarget%20%7B%0A%20%20%20%20%20%20%20%20unsafe%20%7B%20%26*self.0%20%7D%0A%20%20%20%20%7D%0A%7D%0A%0Aimpl%20DerefMut%20for%20BashRef%20%7B%0A%20%20%20%20fn%20deref_mut(%26mut%20self)%20-%3E%20%26mut%20Self%3A%3ATarget%20%7B%0A%20%20%20%20%20%20%20%20unsafe%20%7B%20%26mut%20*self.0%20%7D%0A%20%20%20%20%7D%0A%7D%0A%0Aunsafe%20impl%20std%3A%3Amarker%3A%3ASend%20for%20BashRef%20%7B%7D%0A%0Aconst%20TEST_SIZE%3A%20usize%20%3D%202000%3B%0Afn%20main()%20%7B%0A%20%20%20%20let%20mut%20handle%20%3D%20BashMap%3A%3Anew()%3B%0A%20%20%20%20let%20mut%20handle%20%3D%20handle.handle()%3B%0A%0A%20%20%20%20%2F%2F%20Only%20works%20with%20this%20line.%0A%20%20%20%20%2F%2Fbruh.reserve(200)%3B%0A%0A%20%20%20%20let%20t1%20%3D%20thread%3A%3Aspawn(move%20%7C%7C%20%7B%0A%20%20%20%20%20%20%20%20for%20n%20in%200..TEST_SIZE%20%7B%0A%20%20%20%20%20%20%20%20%20%20%20%20handle.insert(n%20*%202%2C%20n)%3B%0A%20%20%20%20%20%20%20%20%7D%0A%20%20%20%20%7D)%3B%0A%0A%20%20%20%20for%20n%20in%200..TEST_SIZE%20%7B%0A%20%20%20%20%20%20%20%20handle.insert(n%20*%202%20%2B%201%2C%20n)%3B%0A%20%20%20%20%7D%0A%0A%20%20%20%20t1.join().unwrap()%3B%0A%0A%20%20%20%20let%20mut%20failed%20%3D%200%3B%0A%0A%20%20%20%20for%20n%20in%200..TEST_SIZE%20%7B%0A%20%20%20%20%20%20%20%20print!(%22Checking%20%7Bn%7D...%20%22)%3B%0A%20%20%20%20%20%20%20%20let%20read%20%3D%20handle.read()%3B%0A%20%20%20%20%20%20%20%20let%20a%20%3D%20read.get(%26(n%20*%202))%3B%0A%20%20%20%20%20%20%20%20let%20b%20%3D%20read.get(%26(n%20*%202%20%2B%201))%3B%0A%20%20%20%20%20%20%20%20if%20a%20%3D%3D%20Some(%26n)%20%26%26%20b%20%3D%3D%20Some(%26n)%20%7B%0A%20%20%20%20%20%20%20%20%20%20%20%20println!(%22Ok%22)%0A%20%20%20%20%20%20%20%20%7D%20else%20%7B%0A%20%20%20%20%20%20%20%20%20%20%20%20println!(%22Failed%3B%20Found%20%7Ba%3A%3F%7D%2C%20%7Bb%3A%3F%7D%22)%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20failed%20%2B%3D%201%3B%0A%20%20%20%20%20%20%20%20%7D%0A%20%20%20%20%7D%0A%0A%20%20%20%20println!(%22Failed%20%7B%3A.2%7D%25%22%2C%20failed%20as%20f32%20*%20100.%20%2F%20TEST_SIZE%20as%20f32)%3B%0A%7D%0A%0Atrait%20Task%3CT%3A%20Sized%3E%7B%0A%20%20%20%20type%20Input%3A%20Sized%3B%0A%20%20%20%20fn%20hide_par_type(f%3A%20fn(Self%3A%3AInput))%0A%20%20%20%20%20%20%20%20-%3E%20fn%3CT%3E(T)%3B%0A%7D
pub struct Map<'a> {
    nodes: Vec<TaskNode<'a>>,
    indices: Vec<Option<usize>>,
}

impl<'a> Map<'a> {
    fn insert(&mut self, idx: usize, node: TaskNode<'a>) {
        if self.indices.len() <= idx {
            self.reserve(idx - self.indices.len() + 1)
        }

        if let Some(node) = self.indices[idx] {
            let name = "Task name"; // TODO: This could be node.name, if the field is added.
            panic!("Node collision. {name} already exists")
        }

        self.indices[idx] = Some(self.nodes.len());
        self.nodes.push(node)
    }

    fn reserve(&mut self, additional_capacity: usize) {
        self.nodes.reserve(additional_capacity);
        self.indices.append(&mut vec![None; additional_capacity]);
    }

    fn get(&self, idx: usize) -> &TaskNode<'a> {
        &self.nodes[self.indices[idx].unwrap()]
    }

    fn get_mut(&mut self, idx: usize) -> &TaskNode<'a> {
        &mut self.nodes[self.indices[idx].unwrap()]
    }
}

use crate::{branch::Signal, buffer, OptHint, Result, TaskHandle, TaskNode, Work};
use std::{
    future::Future,
    marker::PhantomData,
    sync::mpsc::{channel, Receiver, Sender},
    task::Poll,
};

pub struct Executor<'a> {
    queue: Receiver<Signal<'a>>,
    self_sender: Sender<Signal<'a>>,
    task_graph: Vec<TaskNode<'a>>,
    // TODO: leaf_nodes do a lot of pointles heap allocations
    leaf_nodes: Vec<usize>,
}

impl<'a> Executor<'a> {
    #[inline(always)]
    pub fn new() -> Self {
        let (self_sender, queue) = channel();
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
