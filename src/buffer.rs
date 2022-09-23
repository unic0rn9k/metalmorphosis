use crate::{MorphicIO, Program, Result};
use std::intrinsics::transmute;

pub enum Source<O> {
    Serialized(Vec<u8>),
    Raw(O),
    Uninitialized,
}

impl<O> Source<O> {
    #[inline(always)]
    pub fn alias(&mut self) -> Alias {
        Alias(self as *mut Source<O> as *mut ())
    }

    pub unsafe fn write<T: Program>(&mut self, o: O) -> Result<(), T>
    where
        O: MorphicIO,
    {
        use Source::*;
        match self {
            Serialized(v) => *v = bincode::serialize(&o)?,
            Raw(v) => std::ptr::write(v as *mut O, o),
            Uninitialized => panic!("Cannot write to uninitialized buffer"),
        }
        Ok(())
    }

    pub unsafe fn read<T: Program>(&self) -> Result<O, T>
    where
        O: MorphicIO,
    {
        use Source::*;
        Ok(match self {
            Raw(v) => std::ptr::read(v as *const O),
            Serialized(v) => bincode::deserialize(transmute(&v[..]))?,
            Uninitialized => panic!("Cannot read from uninitialized buffer"),
        })
    }

    #[inline(always)]
    pub fn uninit() -> Self {
        Self::Uninitialized
    }

    pub unsafe fn set_data_format(&mut self, f: char)
    where
        O: MorphicIO,
    {
        match f {
            'r' => *self = Self::Raw(O::buffer()),
            's' => *self = Self::Serialized(vec![]),
            _ => panic!(
                "Please set the data format to either 's' for serialized data or 'r' for raw"
            ),
        }
    }
}

pub struct Alias(*mut ());

impl Alias {
    #[inline(always)]
    pub unsafe fn attach_type<'a, O: MorphicIO>(&self) -> &'a mut Source<O> {
        unsafe { transmute(self.0) }
    }
}

pub const NULL: Source<()> = Source::Uninitialized;
