use std::sync::atomic::{AtomicIsize, Ordering};

use super::DeviceID;

pub struct DeviceOccupancyNode {
    last_thread: AtomicIsize,
    last_mpi: AtomicIsize,
}

impl DeviceOccupancyNode {
    pub fn device(&self) -> Option<DeviceID> {
        let mpi = self.last_mpi.load(Ordering::SeqCst);
        let thread = self.last_thread.load(Ordering::SeqCst);
        if mpi < 0 || thread < 0 {
            None
        } else {
            Some(DeviceID {
                mpi_id: mpi as usize,
                thread_id: thread as usize,
            })
        }
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
