use mpi::{
    point_to_point::ReceiveFuture,
    request::WaitGuard,
    traits::{Communicator, Destination, Equivalence, Source},
};
use serde_derive::{Deserialize, Serialize};
use std::sync::{
    atomic::Ordering,
    mpsc::{channel, Receiver, Sender},
    Arc,
};

use crate::{Graph, Node};

// If node finished, send await to networker
// If node awaited, send Brodcast

// Implementer nu:
// - distribution
// - call mpi time
// - method for getting/setting mpi_instance of node with symbol
//
// - NodeBuilder
// - method 'scheduler' that takes fn(node)->mpi_instance
// - call 'schedular' in Graph on all nodes, in topological ordering, spuriously

pub enum Event {
    Kill,
    AwaitNode(Arc<Node>),
    NodeReady(Arc<Node>, i32),
}

#[derive(Serialize, Deserialize)]
pub enum Message {
    Kill,
    AwaitNode { node: usize },
    NodeReady { data: Vec<u8>, node: usize },
}

impl Message {
    fn serialize(&self) -> Vec<u8> {
        bincode::serialize(&self).unwrap()
    }
    fn deserialize(data: Vec<u8>) -> Self {
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
    pub fn run(&mut self) {
        let size = self.world.size();
        for event in self.events.try_iter() {
            // when we receve await, we need to signal it to graph.

            let (dst, msg) = match event {
                Event::Kill => {
                    for n in 0..size {
                        self.world.process_at_rank(n).send(&Kill.serialize());
                    }
                    return;
                }

                Event::AwaitNode(node) => (
                    node.mpi_instance,
                    AwaitNode {
                        node: node.this_node,
                    }
                    .serialize(),
                ),

                Event::NodeReady(node, rank) => unsafe {
                    (
                        rank,
                        NodeReady {
                            data: (*node.output.get()).serialize(),
                            node: node.this_node,
                        }
                        .serialize(),
                    )
                },
            };

            mpi::request::scope(|scope| {
                let _ = WaitGuard::from(
                    self.world
                        .process_at_rank(dst)
                        .immediate_buffered_send(scope, &msg),
                );
            });
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
