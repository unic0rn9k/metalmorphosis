use super::DeviceID;
use std::sync::{atomic::Ordering, Mutex};

pub struct DeviceOccupancyNode(Mutex<Option<DeviceID>>);

impl DeviceOccupancyNode {
    pub fn device(&self) -> Option<DeviceID> {
        self.0.lock()
    }

    // TODO: Fix me
    pub fn swap(&self, other: &Self) {
        let ord = Ordering::SeqCst;
        let mut mpi = self.last_mpi.load(ord);
        let mut thread = self.last_thread.load(ord);
        mpi = other.last_mpi.swap(mpi, ord);
        thread = other.last_thread.swap(thread, ord);
        self.last_mpi.store(mpi, ord);
        self.last_thread.store(thread, ord);
    }

    pub fn thread_local_swap(&self, other: &Self) {
        let ord = Ordering::SeqCst;
        let mut thread = self.last_thread.load(ord);
        thread = other.last_thread.swap(thread, ord);
        self.last_thread.store(thread, ord);
    }

    pub fn new() -> Self {
        DeviceOccupancyNode {
            last_thread: AtomicIsize::new(-1),
            last_mpi: AtomicIsize::new(-1),
        }
    }

    pub fn from(id: &DeviceID) -> Self {
        DeviceOccupancyNode {
            last_thread: AtomicIsize::new(id.thread_id as isize),
            last_mpi: AtomicIsize::new(id.mpi_id as isize),
        }
    }
}
