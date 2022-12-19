use super::{Pool, ThreadID};
use std::{
    cell::UnsafeCell,
    sync::{Arc, Mutex, MutexGuard},
};

pub struct WorkerStack(UnsafeCell<Option<ThreadID>>);
unsafe impl Sync for WorkerStack {}
pub struct LockedWorkerStack<'a>(MutexGuard<'a, WorkerStack>);

impl<'a> From<&'a Mutex<WorkerStack>> for LockedWorkerStack<'a> {
    fn from(src: &'a Mutex<WorkerStack>) -> Self {
        Self(src.lock().unwrap())
    }
}

pub fn lock(src: &Mutex<WorkerStack>) -> LockedWorkerStack {
    src.into()
}

impl<'a> LockedWorkerStack<'a> {
    pub fn pop(&mut self, pool: &Arc<Pool>) -> Option<ThreadID> {
        if let Some(this) = self.0.device() {
            *self.device() = pool.worker_handles[this.0].prev_unoccupied.clone();
            Some(this)
        } else {
            None
        }
    }

    pub fn insert(&mut self, other: ThreadID, pool: &Arc<Pool>) {
        unsafe {
            (*pool.worker_handles[other.0].prev_unoccupied.0.get()) = self.device().device();
        }
        *self.device() = WorkerStack::from(&other);
    }

    pub fn device(&mut self) -> &mut WorkerStack {
        &mut self.0
    }
}

impl WorkerStack {
    pub fn device(&self) -> Option<ThreadID> {
        unsafe { (*self.0.get()).clone() }
    }

    pub fn new() -> Self {
        WorkerStack(UnsafeCell::new(None))
    }

    pub fn from(id: &ThreadID) -> Self {
        WorkerStack(UnsafeCell::new(Some(id.clone())))
    }

    pub fn clone(&self) -> Self {
        Self(UnsafeCell::new(self.device()))
    }
}
