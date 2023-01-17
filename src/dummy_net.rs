use std::sync::{
    atomic::Ordering,
    mpsc::{channel, Receiver, Sender},
    Arc,
};

use crate::{net::Event, Executor, Graph};

pub struct Networker(Receiver<Event>, Arc<Executor>);

impl Networker {
    pub fn run(&mut self) {
        while self.1.pool.live_threads() != 0 {
            std::hint::spin_loop()
        }
    }
    pub fn rank(&self) -> i32 {
        0
    }
    pub fn kill(self) -> Receiver<Event> {
        self.0
    }
}

pub fn instantiate(graph: Arc<Executor>) -> (Sender<Event>, Networker) {
    let (s, r) = channel();
    (s, Networker(r, graph))
}
