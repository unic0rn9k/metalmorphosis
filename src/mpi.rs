use mpi::traits::{Communicator, Destination, Equivalence, Source};
use serde_derive::{Deserialize, Serialize};
use std::sync::{
    atomic::Ordering,
    mpsc::{channel, Receiver, Sender},
    Arc,
};

use crate::{workpool::ThreadID, Graph, Node};

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
    BrodcastNode(Arc<Node>),
}

#[derive(Serialize, Deserialize)]
pub enum SendableEvent {
    Kill,
    AwaitNode(usize),
    BrodcastNode(Vec<u8>),
}
use SendableEvent::*;

pub struct Networker {
    events: Receiver<Event>,
    universe: mpi::environment::Universe,
    world: mpi::topology::SystemCommunicator,
    graph: Arc<Graph>,
}

impl Networker {
    pub fn run(&mut self) {
        let size = self.world.size();
        for event in self.events.iter() {
            match event {
                Event::Kill => {
                    for n in 0..size {
                        self.world
                            .process_at_rank(n)
                            .send(&bincode::serialize(&Kill).unwrap());
                    }
                    return;
                }

                Event::AwaitNode(node) => {
                    self.world
                        .process_at_rank(node.mpi_instance)
                        .send(&bincode::serialize(&AwaitNode(node.this_node)).unwrap());
                    unsafe {
                        (*node.output.get()).deserialize(
                            self.world
                                .process_at_rank(node.mpi_instance)
                                .receive_vec::<u8>()
                                .0,
                        );
                    }
                    node.done.store(true, Ordering::Release);
                }

                Event::BrodcastNode(node) => unsafe {
                    self.world.process_at_rank(node.mpi_instance).send(
                        &bincode::serialize(&BrodcastNode((*node.output.get()).serialize()))
                            .unwrap(),
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
