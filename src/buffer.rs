use crate::error::{Error, Result};
use std::future::Future;

trait Distributable {
    // Should:
    // - impl mpi::datatype::Equivalence
    //   or
    //   impl Serialize
    // - and impl Sync
}

// TODO: Also needs to take information about what iteration to fetch.
trait Buffer<T> {
    type Writing<'a>: Future<Output = Result<&'a mut T>>
    where
        T: 'a;
    type Reading<'a>: Future<Output = Result<&'a T>>
    where
        T: 'a;

    /// Get a refference for writing a result to the buffer.
    fn get_mut<'a>(&mut self) -> Self::Writing<'a>;
    /// Mark the result as ready for reading. This should lock the buffer, so it will not be written to again.
    fn done(&mut self) -> Result<()>;
    /// Get result from buffer.
    fn get<'a>(&self) -> Self::Reading<'a>;
}

struct WaitAndBleed<T, const W: bool>(*mut TBuffer<T>);

impl<'a, T> Future for WaitAndBleed<T, false> {
    type Output = Result<&T>;

    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        todo!()
    }
}

struct TBuffer<T> {
    ready: bool,
    data: T,
}

impl<T> Buffer<T> for TBuffer<T> {
    type Writing<'a>
    where
        T: 'a;

    type Reading<'a>
    where
        T: 'a;

    fn get_mut<'a>(&mut self) -> Self::Writing<'a> {
        todo!()
    }

    fn done(&mut self) -> Result<()> {
        todo!()
    }

    fn get<'a>(&self) -> Self::Reading<'a> {
        todo!()
    }
}

// TODO: Needs sync (sequential) and async version
struct MultiBuffer<T> {
    data: Vec<TBuffer<T>>,
    iteration: usize,
}
