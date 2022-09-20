use rmp_serde::{decode, encode};
use std::sync::mpsc::{SendError, TryRecvError};

use crate::Program;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error<T: Program> {
    #[error("Error sending data to local executor `{0}`")]
    SendError(SendError<(T, usize, (*mut u8, usize))>),

    #[error("Encode error `{0}`")]
    EncodeError(encode::Error),

    #[error("Decode error `{0}`")]
    DecodeError(decode::Error),

    #[error("Error receiving data from local task `{0}`")]
    ReceiveError(TryRecvError),
}

pub type Result<T, P> = std::result::Result<T, Error<P>>;

macro_rules! impl_convert_err {
    ($from: ty, $to: ident) => {
        impl<T: Program> From<$from> for Error<T> {
            fn from(from: $from) -> Self {
                Self::$to(from)
            }
        }
    };
}

impl_convert_err!(decode::Error, DecodeError);
impl_convert_err!(encode::Error, EncodeError);
impl_convert_err!(SendError<(T, usize, (*mut u8, usize))>, SendError);
impl_convert_err!(TryRecvError, ReceiveError);
