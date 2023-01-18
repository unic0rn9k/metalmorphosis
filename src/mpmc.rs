use std::{
    cell::UnsafeCell,
    fmt::Debug,
    hint::black_box,
    ops::{Deref, DerefMut},
    sync::{
        atomic::{AtomicUsize, Ordering},
        RwLock,
    },
};

// init: [1,2,3]   2
// pop : [1,2, ]   2
// push: [1,2, ,4] 3
// pop : [1,2, ,4] 2
//
//

// If task sets done to true, before polling from awaiters
// then new awaiters will never be pushed while its reading.
// Thus reads can block writes, without consequence.

struct StackSlot<T>(UnsafeCell<Option<T>>);

impl<T: Clone> Clone for StackSlot<T> {
    fn clone(&self) -> Self {
        unsafe { Self(UnsafeCell::new((*self.0.get()).clone())) }
    }
}

pub struct Stack<T> {
    priorities: usize,
    capacity: usize,
    nodes: Vec<StackSlot<T>>,
    next: AtomicUsize,
}
unsafe impl<T> Sync for Stack<T> {}
unsafe impl<T> Send for Stack<T> {}

impl<T: Clone> Stack<T> {
    pub fn fix_capacity(&mut self) {
        self.nodes.append(&mut vec![
            StackSlot(UnsafeCell::new(None));
            self.capacity * self.priorities - self.nodes.len()
        ])
    }

    pub fn push_extend(&mut self, val: T) {
        self.capacity += 1;
        self.nodes.push(StackSlot(UnsafeCell::new(Some(val))));
        self.next.fetch_add(1, Ordering::SeqCst);
    }

    pub fn push(&self, val: T, p: usize) {
        println!("STACK ADDR: {:?}", self as *const _);
        assert!(p < self.priorities);
        let i = self.next.fetch_add(1, Ordering::SeqCst);
        assert!(i < self.capacity, "cap: {}, i: {}", self.capacity, i);
        unsafe { *self.nodes[i + self.capacity * p].0.get() = Some(val) };
    }

    pub fn new(capacity: usize, p: usize) -> Self {
        Self {
            capacity,
            priorities: p,
            nodes: vec![StackSlot(UnsafeCell::new(None)); capacity * p],
            next: AtomicUsize::new(0),
        }
    }

    pub fn undoable(self) -> UndoStack<T> {
        UndoStack {
            stack: self,
            checkpoint: Self::new(0, 0),
        }
    }
}

pub struct StackIter<'a, T>(&'a mut Stack<T>, usize);

impl<'a, T> IntoIterator for &'a mut Stack<T> {
    type Item = T;

    type IntoIter = StackIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        StackIter(
            self,
            self.next.load(Ordering::SeqCst) + self.capacity * (self.priorities - 1),
        )
    }
}

impl<'a, T> Iterator for StackIter<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.1 == 0 {
            return None;
        }
        self.1 -= 1;
        match self.0.nodes[self.1].0.get_mut().take() {
            None => self.next(),
            some => some,
        }
    }
}

impl<'a, T> Drop for StackIter<'a, T> {
    fn drop(&mut self) {
        self.0
            .next
            .store(self.1 % self.0.capacity, Ordering::SeqCst)
    }
}

impl<T: Clone> Clone for Stack<T> {
    fn clone(&self) -> Self {
        Self {
            priorities: self.priorities,
            capacity: self.capacity,
            nodes: self.nodes.clone(),
            next: AtomicUsize::new(self.next.load(Ordering::SeqCst)),
        }
    }
}

impl<T: Debug> Debug for &mut Stack<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Stack")
            .field(
                "nodes",
                &self
                    .nodes
                    .iter()
                    .enumerate()
                    .filter_map(|(n, s)| unsafe { (*s.0.get()).as_ref().map(|s| (n, s)) })
                    .collect::<Vec<_>>(),
            )
            .field("next", &self.next)
            .finish()
    }
}

/// A que that can be reverted to a previous state.
pub struct UndoStack<T: Clone> {
    stack: Stack<T>,
    checkpoint: Stack<T>,
}

impl<T: Clone> UndoStack<T> {
    /// Revert que to last checkpoint.
    pub fn undo(&mut self) {
        self.stack = self.checkpoint.clone()
    }
    /// Save current state of que as checkpoint.
    pub fn checkpoint(&mut self) {
        self.checkpoint = self.stack.clone()
    }
}

impl<T: Clone> Deref for UndoStack<T> {
    type Target = Stack<T>;

    fn deref(&self) -> &Self::Target {
        &self.stack
    }
}

impl<T: Clone> DerefMut for UndoStack<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.stack
    }
}

extern crate test;
use test::Bencher;

#[bench]
fn stack(b: &mut Bencher) {
    b.iter(|| {
        let mut stack = Stack::new(10, 3);
        stack.push(black_box(2usize), 2);
        black_box(stack.into_iter().next().unwrap());
    })
}

#[test]
fn stack_par() {
    use std::{sync::Arc, thread};

    let size = 10;
    let p = 3;

    let first = Arc::new(RwLock::new(Stack::new(size, p)));
    let second = first.clone();

    thread::spawn(move || {
        for a in 0..size {
            first.read().unwrap().push(a, a % 3)
        }
    })
    .join()
    .unwrap();

    thread::spawn(move || {
        for (a, b) in second.write().unwrap().into_iter().enumerate() {
            //assert_eq!(9 - a, b)
            println!("{a}: {b}");
            if a >= 4 {
                break;
            }
        }
        for (a, b) in second.write().unwrap().into_iter().enumerate() {
            //assert_eq!(9 - a, b)
            println!("{a}: {b}");
        }
    })
    .join()
    .unwrap();
}
