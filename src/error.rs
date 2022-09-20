use rmp_serde::{decode, encode};
use std::sync::mpsc::{SendError, TryRecvError};

use crate::Program;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error<T: Program> {
    #[error("Error sending data to local executor `{0}`")]
    SendError(SendError<(T, usize, (*mut u8, usize))>),

    #[error(transparent)]
    EncodeError(#[from] encode::Error),

    #[error(transparent)]
    DecodeError(#[from] decode::Error),

    #[error(transparent)]
    ReceiveError(#[from] TryRecvError),
}

impl<T: Program> From<SendError<(T, usize, (*mut u8, usize))>> for Error<T> {
    fn from(v: SendError<(T, usize, (*mut u8, usize))>) -> Self {
        Self::SendError(v)
    }
}

pub type Result<T, P> = std::result::Result<T, Error<P>>;
