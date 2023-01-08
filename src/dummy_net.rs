use std::sync::{
    atomic::Ordering,
    mpsc::{channel, Receiver, Sender},
    Arc,
};

use crate::{net::Event, Graph};

pub struct Networker(Receiver<Event>, Arc<Graph>);

impl Networker {
    pub fn run(&mut self) {
        while !self.1.nodes[0].done.load(Ordering::SeqCst) {
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

pub fn instantiate(graph: Arc<Graph>) -> (Sender<Event>, Networker) {
    let (s, r) = channel();
    (s, Networker(r, graph))
}
