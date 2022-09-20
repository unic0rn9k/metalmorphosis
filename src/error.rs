use bincode;
use std::sync::mpsc::{SendError, TryRecvError};

use crate::{OutputSlice, Program};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error<T: Program> {
    #[error("Error sending data to local executor `{0}`")]
    SendError(SendError<(T, usize, OutputSlice)>),

    #[error(transparent)]
    BincodeError(#[from] bincode::Error),

    #[error(transparent)]
    ReceiveError(#[from] TryRecvError),
}

impl<T: Program> From<SendError<(T, usize, OutputSlice)>> for Error<T> {
    fn from(v: SendError<(T, usize, OutputSlice)>) -> Self {
        Self::SendError(v)
    }
}

pub type Result<T, P> = std::result::Result<T, Error<P>>;
