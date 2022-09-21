use bincode;
use std::sync::mpsc::{SendError, TryRecvError};

use crate::{Program, Signal};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error<T: Program> {
    #[error("Error sending data to local executor `{0}`")]
    SendError(SendError<Signal<T>>),

    #[error(transparent)]
    BincodeError(#[from] bincode::Error),

    #[error(transparent)]
    ReceiveError(#[from] TryRecvError),
}

impl<T: Program> From<SendError<Signal<T>>> for Error<T> {
    fn from(v: SendError<Signal<T>>) -> Self {
        Self::SendError(v)
    }
}

pub type Result<T, P> = std::result::Result<T, Error<P>>;
