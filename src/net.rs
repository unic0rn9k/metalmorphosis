use mpi::{
    point_to_point::{ReceiveFuture, Status},
    request::{CancelGuard, WaitGuard},
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

// TODO: Should contain clones of node.awaiter
//       Also make sure awaiters aren't cloned in parallel
pub struct Networker {
    events: Receiver<Event>,
    universe: mpi::environment::Universe,
    world: mpi::topology::SystemCommunicator,
    graph: Arc<Graph>,
}

impl Networker {
    pub fn run(self) -> Receiver<Event> {
        let size = self.world.size();
        loop {
            let mut buffer = [0u8; 1024];
            let kill_network = mpi::request::scope(|scope| {
                let mut external_event = self
                    .world
                    .any_process()
                    .immediate_receive_into(scope, &mut buffer);

                loop {
                    if let Ok(internal_event) = self.events.try_recv() {
                        let (dst, msg) = match internal_event {
                            Event::Kill => {
                                for n in 0..size {
                                    self.world.process_at_rank(n).send(&Kill.serialize());
                                }
                                external_event.wait_without_status();
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
                            return self
                                .handle_external_event(Message::deserialize(&data[0..bytes]));
                        }
                        Err(same) => same,
                    }
                }
            });
            if kill_network {
                return self.events;
            };
        }
    }
    pub fn rank(&self) -> i32 {
        self.world.rank()
    }

    fn handle_external_event(&self, msg: Message) -> bool {
        match msg {
            Kill => return true,
            AwaitNode { awaited, awaiter } => {
                println!(
                    "{} ::NET:: {} awaited {}",
                    self.graph.mpi_instance(),
                    self.graph.nodes[awaiter].name,
                    self.graph.nodes[awaited].name,
                );
                let node = &self.graph.nodes[awaited];

                while node.is_being_polled.swap(true, Ordering::Acquire) {}
                // FIXME: Awaiter gets polled in assign bellow,
                //        which runs this again, thus infinite loop.
                //        (node = awaited)
                node.awaiter.send(awaiter).unwrap();
                node.is_being_polled.store(false, Ordering::Release);

                if node.done.load(Ordering::Acquire) {
                    self.graph.pool.assign([self.graph.nodes[awaited].clone()])
                }
            }
            NodeReady { data, node } => {
                let node = self.graph.nodes[node].clone();
                if node.is_being_polled.swap(true, Ordering::Acquire) {
                    panic!("NodeReady event for in-use node: {}", node.name);
                }

                node.done.store(true, Ordering::Release);
                unsafe { (*node.output.get()).deserialize(data) }

                self.graph.print();
                self.graph.compute(node.this_node);
                println!(
                    "{} ::NET:: node {} ready",
                    self.graph.mpi_instance(),
                    node.name
                );
            }
        }
        false
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
