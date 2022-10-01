use crate::{Program, Signal};
use std::sync::mpsc::{SendError, TryRecvError};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error<'a, T: Program<'a>> {
    #[error("Error sending data to local executor `{0}`")]
    SendError(SendError<Signal<'a, T>>),

    #[error(transparent)]
    BincodeError(#[from] bincode::Error),

    #[error(transparent)]
    ReceiveError(#[from] TryRecvError),
}

impl<'a, T: Program<'a>> From<SendError<Signal<'a, T>>> for Error<'a, T> {
    fn from(v: SendError<Signal<'a, T>>) -> Self {
        Self::SendError(v)
    }
}

pub type Result<'a, T, P> = std::result::Result<T, Error<'a, P>>;
