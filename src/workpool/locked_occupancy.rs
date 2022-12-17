use super::{DeviceID, Pool};
use std::{
    cell::UnsafeCell,
    sync::{Arc, Mutex, MutexGuard},
};

pub struct OccupancyNode(UnsafeCell<Option<DeviceID>>);
unsafe impl Sync for OccupancyNode {}
pub struct LockedOccupancyNode<'a>(MutexGuard<'a, OccupancyNode>);

impl<'a> From<&'a Mutex<OccupancyNode>> for LockedOccupancyNode<'a> {
    fn from(src: &'a Mutex<OccupancyNode>) -> Self {
        Self(src.lock().unwrap())
    }
}

pub fn lock(src: &Mutex<OccupancyNode>) -> LockedOccupancyNode {
    src.into()
}

impl<'a> LockedOccupancyNode<'a> {
    pub fn pop(&mut self, pool: &Arc<Pool>) -> Option<DeviceID> {
        if let Some(this) = self.0.device().map(|s| s) {
            // This does not need to lock the mutex, since it will always be unlocked.
            let ret = this.clone();
            *self.device() = pool.worker_handles[this.thread_id].prev_unoccupied.clone();
            Some(ret)
        } else {
            None
        }
    }

    pub fn insert(&mut self, other: DeviceID, pool: &Arc<Pool>) {
        unsafe {
            (*pool.worker_handles[other.thread_id].prev_unoccupied.0.get()) =
                self.device().device();
        }
        *self.device() = OccupancyNode::from(&other);
    }

    pub fn device(&mut self) -> &mut OccupancyNode {
        &mut self.0
    }
}

impl OccupancyNode {
    pub fn device(&self) -> Option<DeviceID> {
        unsafe { (*self.0.get()).clone() }
    }

    pub fn new() -> Self {
        OccupancyNode(UnsafeCell::new(None))
    }

    pub fn from(id: &DeviceID) -> Self {
        OccupancyNode(UnsafeCell::new(Some(id.clone())))
    }

    pub fn clone(&self) -> Self {
        Self(UnsafeCell::new(self.device()))
    }
}
