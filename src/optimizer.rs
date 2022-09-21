use crate::{Program, TaskNode};
use std::{marker::PhantomData, sync::Arc};

pub struct Optimizer<T: Program> {
    // Table of benchmark data
    phantom_data: PhantomData<T>,
}

impl<T: Program> Optimizer<T> {
    pub fn select_next_task(&self, _: &Vec<Arc<TaskNode<T>>>) -> usize {
        todo!()
    }

    pub fn hint(&self, _: T) -> HintFromOptimizer {
        HintFromOptimizer {
            fast_and_unsafe_serialization: true,
        }
    }

    pub fn new() -> Self {
        Self {
            phantom_data: PhantomData,
        }
    }
}

#[derive(Clone, Copy)]
pub struct HintFromOptimizer {
    pub fast_and_unsafe_serialization: bool,
}

pub fn main_hint() -> HintFromOptimizer {
    HintFromOptimizer {
        fast_and_unsafe_serialization: true,
    }
}
