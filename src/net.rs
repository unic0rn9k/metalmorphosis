use mpi::{
    point_to_point::ReceiveFuture,
    traits::{Communicator, Destination, Equivalence, Source},
};
use serde_derive::{Deserialize, Serialize};
use std::sync::{
    atomic::Ordering,
    mpsc::{channel, Receiver, Sender},
    Arc,
};

use crate::{Graph, Node};

pub enum Event {
    Kill,
    AwaitNode(Arc<Node>),
    BrodcastNode(Arc<Node>, i32),
}

#[derive(Serialize, Deserialize)]
pub enum Message {
    Kill,
    AwaitNode { node: usize, reader: i32 },
    BrodcastNode { data: Vec<u8>, node: usize },
}

impl Message {
    fn send(&self) -> Vec<u8> {
        bincode::serialize(&self).unwrap()
    }
    fn recv(data: Vec<u8>) -> Self {
        bincode::deserialize(&data).expect("Failed to deserialize network event")
    }
}

use Message::*;

pub struct Networker {
    events: Receiver<Event>,
    universe: mpi::environment::Universe,
    world: mpi::topology::SystemCommunicator,
    graph: Arc<Graph>,
}

impl Networker {
    pub fn run(mut self) {
        return;
        let size = self.world.size();
        for event in self.events.try_iter() {
            match event {
                Event::Kill => {
                    for n in 0..size {
                        self.world.process_at_rank(n).send(&Kill.send());
                    }
                    return;
                }

                Event::AwaitNode(node) => {
                    self.world.process_at_rank(node.mpi_instance).send(
                        &AwaitNode {
                            node: node.this_node,
                            reader: self.world.rank(),
                        }
                        .send(),
                    );
                }

                Event::BrodcastNode(node, rank) => unsafe {
                    self.world.process_at_rank(rank).send(
                        &BrodcastNode {
                            data: (*node.output.get()).serialize(),
                            node: node.this_node,
                        }
                        .send(),
                    )
                },
            }
        }
    }
    pub fn rank(&self) -> i32 {
        self.world.rank()
    }
}

pub fn instantiate(graph: Arc<Graph>) -> (Sender<Event>, Networker) {
    let (universe, threading) = mpi::initialize_with_threading(mpi::Threading::Funneled).unwrap();
    if threading != mpi::Threading::Funneled {
        panic!("Only supports funneled mpi threading");
    }
    let world = universe.world();
    let (s, r) = channel();
    (
        s,
        Networker {
            events: r,
            universe,
            world,
            graph,
        },
    )
}
