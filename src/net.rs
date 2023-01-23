use mpi::{
    environment::Universe,
    request::WaitGuard,
    traits::{AsDatatype, Communicator, Destination, Source},
};
use serde_derive::{Deserialize, Serialize};
use std::sync::{
    atomic::Ordering,
    mpsc::{channel, sync_channel, Receiver, Sender},
    Arc,
};

use crate::{Graph, DEBUG};

// TODO: Make networking a task.
// Strictly it would only need to be polled once a node has finished.
// In that case, it would be most efficient to read all external events, before acting on internal events.

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
    AwaitNode { awaited: usize },
    NodeDone { awaited: usize },
    Consumes { awaited: usize, at: i32 },
}

#[derive(Serialize, Deserialize)]
pub enum Message {
    Kill,
    AwaitNode { awaited: usize },
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
    _universe: Universe,
    world: mpi::topology::SystemCommunicator,
    graph: Arc<Graph>,
    awaited_at: Vec<Vec<i32>>,
}

impl Networker {
    pub fn run(&mut self) {
        println!("running net...");
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

                            Event::AwaitNode { awaited } => (
                                std::slice::from_ref(&self.graph.nodes[awaited].mpi_instance),
                                AwaitNode { awaited }.serialize(),
                            ),

                            Event::NodeDone { awaited } => unsafe {
                                println!("{awaited} done at {}", self.world.rank());
                                let node = &self.graph.nodes[awaited];
                                (
                                    &self.awaited_at[awaited][..],
                                    NodeReady {
                                        data: (*node.output.get()).serialize(),
                                        node: node.this_node,
                                    }
                                    .serialize(),
                                )
                            },

                            Event::Consumes { awaited, at } => {
                                println!("C... {awaited} awaited at {at}");
                                self.awaited_at[awaited].push(at);
                                continue;
                            }
                        };

                        mpi::request::scope(|scope| {
                            for dst in dst {
                                println!("Sending results to {dst}");
                                let _ = WaitGuard::from(
                                    self.world.process_at_rank(*dst).immediate_send(scope, &msg),
                                );
                            }
                        });
                    }

                    external_event = match external_event.test_with_data() {
                        Ok((msg, data)) => {
                            let bytes = msg.count(data.as_datatype()) as usize;
                            return self.handle_external_event(
                                Message::deserialize(&data[0..bytes]),
                                msg.source_rank(),
                            );
                        }
                        Err(same) => same,
                    }
                }
            });
            if kill_network {
                println!("Exiting net...");
                return;
            };
        }
    }
    pub fn kill(self) -> Receiver<Event> {
        self.events
    }
    pub fn rank(&self) -> i32 {
        self.world.rank()
    }

    fn handle_external_event(&mut self, msg: Message, src: i32) -> bool {
        match msg {
            Kill => return true,
            AwaitNode { awaited } => {
                println!(
                    "mpi:{} awaited {}",
                    self.graph.mpi_instance(),
                    self.graph.nodes[awaited].name,
                );
                self.awaited_at[awaited].push(src);

                self.graph.pool.assign([&self.graph.nodes[awaited]]);
            }
            NodeReady { data, node } => {
                println!("mpi:{} external ready {}", self.graph.mpi_instance(), node);
                let node = self.graph.nodes[node].clone();
                //if node.is_being_polled.swap(true, Ordering::Acquire) {
                //    panic!("NodeReady event for in-use node: {}", node.name);
                //}
                while !node.try_poll("net") {}

                node.done.store(true, Ordering::Release);
                unsafe { (*node.output.get()).deserialize(&data) }

                //self.graph.print();
                self.graph.assign_children_of(&node);
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
            _universe: universe,
            awaited_at: vec![vec![]; graph.nodes.len()],
            events: r,
            world,
            graph,
        },
    )
}
