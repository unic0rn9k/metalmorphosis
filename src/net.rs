use mpi::{
    point_to_point::{ReceiveFuture, Status},
    request::WaitGuard,
    traits::{AsDatatype, Communicator, Destination, Equivalence, Source},
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

// Dont need seperate Message and Event.
// When send NodeReady, just serialize node.output once pr recipient.
pub enum Event {
    Kill,
    AwaitNode { awaited: usize, reader: usize },
    NodeDone { awaited: usize, reader: usize },
}

#[derive(Serialize, Deserialize)]
pub enum Message {
    Kill,
    AwaitNode { awaited: usize, awaiter: usize },
    NodeReady { data: Vec<u8>, node: usize },
}

impl Message {
    fn serialize(&self) -> Vec<u8> {
        bincode::serialize(&self).unwrap()
    }
    fn deserialize(data: &[u8]) -> Self {
        bincode::deserialize(data).expect("Failed to deserialize network event")
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
        loop {
            let mut buffer = [0u8; 1024];
            let kill_network = mpi::request::scope(|scope| {
                let mut external_event = self
                    .world
                    .any_process()
                    .immediate_receive_into(scope, &mut buffer);

                loop {
                    for internal_event in self.events.try_iter() {
                        let (dst, msg) = match internal_event {
                            Event::Kill => {
                                for n in 0..size {
                                    self.world.process_at_rank(n).send(&Kill.serialize());
                                }
                                return true;
                            }

                            Event::AwaitNode {
                                awaited,
                                reader: awaiter,
                            } => (
                                self.graph.nodes[awaited].mpi_instance,
                                AwaitNode { awaited, awaiter }.serialize(),
                            ),

                            Event::NodeDone { awaited, reader } => unsafe {
                                let node = &self.graph.nodes[awaited];
                                (
                                    self.graph.nodes[reader].mpi_instance,
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
                                self.world.process_at_rank(dst).immediate_send(scope, &msg),
                            );
                        });
                    }

                    external_event = match external_event.test_with_data() {
                        Ok((msg, data)) => {
                            let bytes = msg.count(data.as_datatype()) as usize;
                            let sender = msg.source_rank();
                            self.handle_external_event(Message::deserialize(&data[0..bytes]));
                            return false;
                        }
                        Err(same) => same,
                    }
                }
            });
            if kill_network {
                return;
            };
        }
    }
    pub fn rank(&self) -> i32 {
        self.world.rank()
    }

    fn handle_external_event(&self, msg: Message) {
        match msg {
            Kill => panic!("Just died in your arms tonight"),
            AwaitNode { awaited, awaiter } => println!(
                "{} ::NET:: {awaiter} awaited {awaited}",
                self.graph.mpi_instance()
            ),
            NodeReady { data, node } => {
                println!("{} ::NET:: Some node finished", self.graph.mpi_instance())
            }
        }
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
