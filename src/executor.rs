use crate::*;

pub struct Executor<T: Program> {
    queue: Receiver<(T, usize, OutputSlice)>,
    self_sender: SyncSender<(T, usize, OutputSlice)>,
    task_graph: Vec<Arc<TaskNode<T>>>,
}

impl<T: Program> Executor<T> {
    #[inline(always)]
    pub fn new() -> Self {
        let (self_sender, queue) = sync_channel(1000);
        Self {
            queue,
            self_sender,
            task_graph: vec![],
        }
    }

    pub fn run(&mut self, main: T) -> Result<(), T> {
        self.branch(main, 0, &mut vec![] as *mut Vec<u8>);
        let mut n = 0;

        'polling: loop {
            // Poll
            if n == self.task_graph.len() {
                n = 0;
                println!("Ran out of nodes. Fetching new ones...");
                let mut branch = self.queue.try_recv();
                while branch.is_ok() {
                    let (token, parent, output) = unsafe { branch.unwrap_unchecked() };
                    self.branch(token, parent, output);
                    branch = self.queue.try_recv();
                }
            }

            if self.task_graph[n].children != 0 {
                n += 1;
                continue 'polling;
            }

            println!(
                "Polling node {:?} of {}",
                self.task_graph[n].token,
                self.task_graph.len()
            );
            if unsafe { Arc::get_mut_unchecked(&mut self.task_graph[n]) }
                .poll()
                .is_ready()
            {
                println!("Finished node {:?}", self.task_graph[n].token);
                if self.task_graph[n].this_node == self.task_graph[n].parent {
                    println!(
                        "DONE! {} nodes left behind: {:?}",
                        self.task_graph.len(),
                        self.task_graph.iter().map(|n| n.token).collect::<Vec<_>>()
                    );
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

            println!("  ... [{:?}]\n", self.task_graph[n].token);

            n += 1;
        }
    }

    pub fn branch(&mut self, token: T, parent: usize, output: OutputSlice) {
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
            token,
            children: 0,
        };
        self.task_graph.push(Arc::new(node));
        let last = self.task_graph.len() - 1;
        let node = &mut self.task_graph[last];
        let tmp = Box::new(token.future(Arc::clone(node)));
        unsafe { Arc::get_mut_unchecked(node) }.future = tmp;
    }
}
