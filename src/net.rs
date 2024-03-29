use mpi::{
    environment::Universe,
    request::WaitGuard,
    topology::SystemCommunicator,
    traits::{AsDatatype, Communicator, Destination, Source},
};
use serde_derive::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::{
        atomic::Ordering,
        mpsc::{channel, Receiver, Sender},
        Arc,
    },
};

use crate::{Executor, Graph, NodeId, DEBUG};

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
    AwaitNode { awaited: NodeId },
    NodeDone { awaited: NodeId },
    Consumes { awaited: NodeId, at: i32 },
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
    world: SystemCommunicator,
    graph: Arc<Executor>,
    awaited_at: HashMap<usize, Vec<i32>>,
    awaited: HashMap<usize, NodeId>,
}

pub struct NetHandle {
    net: Sender<Event>,
    graph: Arc<Executor>,
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
                    if let Ok(internal_event) = self.events.try_recv() {
                        let at;
                        let (dst, msg) = match internal_event {
                            Event::Kill => {
                                for n in 0..size {
                                    self.world.process_at_rank(n).send(&Kill.serialize());
                                }
                                external_event.wait_without_status();
                                return true;
                            }

                            Event::AwaitNode { awaited } => {
                                at = awaited.mpi_instance;
                                self.awaited.insert(awaited.this_node, awaited.clone());
                                (
                                    std::slice::from_ref(&at),
                                    AwaitNode {
                                        awaited: awaited.this_node,
                                    }
                                    .serialize(),
                                )
                            }

                            Event::NodeDone { awaited } => unsafe {
                                println!("net event: {awaited:?} done");
                                let node = awaited.this_node;
                                (
                                    match self.awaited_at.get(&node) {
                                        Some(awaiters) => &awaiters[..],
                                        _ => continue,
                                    },
                                    NodeReady {
                                        data: awaited.output.serialize(),
                                        node,
                                    }
                                    .serialize(),
                                )
                            },

                            Event::Consumes { awaited, at } => {
                                println!("net event: {awaited:?} awaited");
                                self.awaited_at
                                    .entry(awaited.this_node)
                                    .or_insert(vec![])
                                    .push(at);
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
                if DEBUG {
                    println!(
                        "net event: #{} awaited @{}",
                        awaited,
                        self.graph.mpi_instance(),
                    )
                };
                self.awaited_at.entry(awaited).or_insert(vec![]).push(src);
                //self.graph.pool.assign([awaited])
            }
            NodeReady { data, node } => {
                let node = self.awaited[&node].clone();
                if node.is_being_polled.swap(true, Ordering::SeqCst) {
                    panic!("NodeReady event for in-use node: {}", node.name);
                }

                node.done.store(true, Ordering::SeqCst);
                unsafe { node.output.deserialize(&data) }

                //self.graph.print();
                self.graph.assign_children_of(&node);
                if DEBUG {
                    println!(
                        "{} ::NET:: node {} ready",
                        self.graph.mpi_instance(),
                        node.name
                    )
                };
            }
        }
        false
    }
}

pub fn instantiate(graph: Arc<Executor>) -> (Sender<Event>, Networker) {
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
            awaited_at: HashMap::new(),
            events: r,
            world,
            graph,
            awaited: HashMap::new(),
        },
    )
}
