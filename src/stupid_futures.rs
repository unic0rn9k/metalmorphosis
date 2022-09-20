use std::{future::Future, task::Poll};

pub struct HaltOnceWaker(bool);

/// A future that always returns Poll::Ready()
impl Future for HaltOnceWaker {
    type Output = ();

    #[inline(always)]
    fn poll(
        mut self: std::pin::Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        if self.0 {
            Poll::Ready(())
        } else {
            self.as_mut().0 = true;
            Poll::Pending
        }
    }
}

pub fn halt_once() -> HaltOnceWaker {
    HaltOnceWaker(false)
}

/// A future that panics if you poll it.
pub struct UninitFuture;

impl Future for UninitFuture {
    type Output = ();

    fn poll(
        self: std::pin::Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        panic!("Attempted to poll UninitFuture");
    }
}
