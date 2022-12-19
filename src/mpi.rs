use mpi::traits::Communicator;
use std::sync::mpsc::{channel, Receiver, Sender};

// If node finished, send await to networker
// If node awaited, send Brodcast

pub enum Event {
    AwaitTask,
    Brodcast,
}

pub struct Networker {
    events: Receiver<Event>,
    universe: mpi::environment::Universe,
    world: mpi::topology::SystemCommunicator,
}

impl Networker {
    pub fn run(&mut self) {}
    pub fn rank(&self) -> i32 {
        self.world.rank()
    }
}

pub fn instantiate() -> (Sender<Event>, Networker) {
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
        },
    )
}
