//use crate::branch::Signal;
use std::sync::mpsc::{TryRecvError};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    //    #[error("Error sending data to local executor `{0}`")]
    //    SendError(SendError<Signal<'a>>),
    #[error(transparent)]
    Bincode(#[from] bincode::Error),

    #[error(transparent)]
    Receive(#[from] TryRecvError),

    #[error(transparent)]
    StdIO(#[from] std::io::Error),

    #[error("It is illegal to call `TaskHandle::output` from main")]
    MainWrite,

    #[error("Name collision: '{0}' already exists")]
    NameCollision(String),

    #[error("Node '{0}' not found")]
    UnknownNode(String),
}

//impl<'a> From<SendError<Signal<'a>>> for Error<'a> {
//    fn from(v: SendError<Signal<'a>>) -> Self {
//        Self::SendError(v)
//    }
//}

pub type Result<T> = std::result::Result<T, Error>;
