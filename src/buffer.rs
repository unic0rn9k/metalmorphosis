use crate::{MorphicIO, Program, Result};
use std::{any::type_name, intrinsics::transmute};

pub enum Source<O> {
    Serialized(Vec<u8>),
    Raw(O),
    Uninitialized,
    Const,
}

impl<O> Source<O> {
    #[inline(always)]
    pub fn alias(&mut self) -> Alias {
        Alias(self as *mut Source<O> as *mut ())
    }

    pub fn write<T: Program>(&mut self, o: O) -> Result<(), T>
    where
        O: MorphicIO,
    {
        use Source::*;
        match self {
            Serialized(v) => *v = bincode::serialize(&o)?,
            Raw(v) => *v = o,
            _ => panic!("Cannot write to uninitialized or const buffer"),
        }
        Ok(())
    }

    pub fn read<T: Program>(self) -> Result<O, T>
    where
        O: MorphicIO,
    {
        use Source::*;
        Ok(match self {
            Raw(v) => v,
            Serialized(v) => unsafe { bincode::deserialize(transmute(&v[..]))? },
            _ => panic!("Cannot read from uninitialized or const buffer"),
        })
    }

    #[inline(always)]
    pub fn uninit() -> Self {
        Self::Uninitialized
    }

    fn fmt(&self) -> &'static str {
        match self {
            Source::Serialized(_) => "serialized",
            Source::Raw(_) => "raw",
            Source::Uninitialized => "uninitialized",
            Source::Const => "const",
        }
    }

    fn is_const(&self) -> bool {
        match self {
            Source::Const => true,
            _ => false,
        }
    }

    pub unsafe fn set_data_format<const FORMAT: char>(&mut self)
    where
        O: MorphicIO,
    {
        match FORMAT {
            'r' if O::IS_COPY && !self.is_const() => *self = Self::Raw(O::buffer()),
            's' if !self.is_const() => *self = Self::Serialized(vec![]),
            _ => panic!(
                "Tried to set a {} buffer, of type `{}`, to '{FORMAT}'",
                self.fmt(),
                type_name::<O>()
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

pub const NULL: Source<()> = Source::Const;
