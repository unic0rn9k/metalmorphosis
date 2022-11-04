// TODO: Merge me into buffer!

use std::{any::type_name, mem::size_of};

use crate::buffer;

pub trait Allocator {
    fn alloc<T: MorphicIO<'a>>(&mut self) -> buffer::Alias<'a>;
    fn new(capacity_in_bytes: usize) -> Self;
}

pub trait Buffer {
    fn slice(&self, a: usize, b: usize) -> &[u8];
    fn slice_mut(&mut self, a: usize, b: usize) -> &mut [u8];
}

impl<const LEN: usize> Buffer for [u8; LEN] {
    fn slice(&self, a: usize, b: usize) -> &[u8] {
        &self[a..b]
    }

    fn slice_mut(&mut self, a: usize, b: usize) -> &mut [u8] {
        &mut [a..b]
    }
}

impl Buffer for Vec<u8> {
    fn slice(&self, a: usize, b: usize) -> &[u8] {
        &self[a..b]
    }

    fn slice_mut(&mut self, a: usize, b: usize) -> &mut [u8] {
        &mut [a..b]
    }
}

pub struct Ring<T: Buffer> {
    memory: T,
    idx: AtomicUsize,
}

impl<T: Buffer> Allocator for Ring<T> {
    fn alloc<T: MorphicIO<'a>>(&mut self) -> buffer::Alias<'a> {
        todo!()
    }

    fn new(capacity_in_bytes: usize) -> Self {
        todo!()
    }
}

impl<'a, T> Allocator for buffer::Source<'a, T> {
    fn alloc<T: MorphicIO<'a>>(&mut self) -> buffer::Alias<'a> {
        self.alias()
    }

    fn new(capacity_in_bytes: usize) -> Self {
        if capacity_in_bytes != size_of::<T>() {
            panic!(
                "{} is only a valid allocator, if its capacity is {} (size of {})",
                type_name::<buffer::Source<'a, T>>(),
                size_of::<T>(),
                type_name::<T>()
            )
        }
    }
}
