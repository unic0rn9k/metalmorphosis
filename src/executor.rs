use crate::*;

pub struct Executor<T: Program> {
    queue: Receiver<Signal<T>>,
    self_sender: SyncSender<Signal<T>>,
    task_graph: Vec<Arc<TaskNode<T>>>,
    optimizer: optimizer::Optimizer<T>,
}

impl<T: Program> Executor<T> {
    #[inline(always)]
    pub fn new() -> Self {
        let (self_sender, queue) = sync_channel(1000);
        Self {
            queue,
            self_sender,
            task_graph: vec![],
            optimizer: optimizer::Optimizer::new(),
        }
    }

    pub fn run(&mut self, main: T) -> Result<(), T> {
        self.branch(Signal::Branch {
            token: main,
            parent: 0,
            output: OutputSlice {
                vec: &mut vec![] as *mut Vec<u8>,
            },
            optimizer_hint: optimizer::main_hint(),
        });
        let mut n = 0;

        'polling: loop {
            // Poll
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

            if unsafe { Arc::get_mut_unchecked(&mut self.task_graph[n]) }
                .poll()
                .is_ready()
            {
                if self.task_graph[n].this_node == self.task_graph[n].parent {
                    self.task_graph.clear();
                    return Ok(());
                } else {
                    let parent = self.task_graph[n].parent;
                    self.task_graph.remove(n);
                    n = parent;
                    unsafe { Arc::get_mut_unchecked(&mut self.task_graph[n]).children -= 1 };
                    continue 'polling;
                }
            }

            println!(":---- {n}\n:");

            n += 1;
        }
    }

    pub fn branch(&mut self, branch: Signal<T>) {
        let Signal::Branch {
            parent,
            output,
            token,
            optimizer_hint,
        } = branch else{todo!()};

        match self.task_graph.get_mut(parent) {
            None => {}
            Some(parent) => unsafe { Arc::get_mut_unchecked(parent).children += 1 },
        }

        let node = TaskNode {
            sender: self.self_sender.clone(),
            output,
            future: Box::new(UninitFuture),
            parent,
            this_node: self.task_graph.len(),
            children: 0,
            optimizer: &self.optimizer as *const optimizer::Optimizer<T>,
            optimizer_hint,
        };

        self.task_graph.push(Arc::new(node));
        let last = self.task_graph.len() - 1;
        let node = &mut self.task_graph[last];
        let tmp = Box::new(token.future(Arc::clone(node)));
        unsafe { Arc::get_mut_unchecked(node) }.future = tmp;
    }
}
