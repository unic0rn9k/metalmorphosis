use crate::branch::Signal;
use std::sync::mpsc::{SendError, TryRecvError};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error<'a> {
    #[error("Error sending data to local executor `{0}`")]
    SendError(SendError<Signal<'a>>),

    #[error(transparent)]
    BincodeError(#[from] bincode::Error),

    #[error(transparent)]
    ReceiveError(#[from] TryRecvError),

    #[error(transparent)]
    StdIO(#[from] std::io::Error),

    #[error("It is illegal to call `TaskHandle::output` from main")]
    MainWrite,
}

impl<'a> From<SendError<Signal<'a>>> for Error<'a> {
    fn from(v: SendError<Signal<'a>>) -> Self {
        Self::SendError(v)
    }
}

pub type Result<'a, T> = std::result::Result<T, Error<'a>>;
