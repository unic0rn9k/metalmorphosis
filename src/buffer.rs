use crate::{MorphicIO, Result};
use std::{any::type_name, intrinsics::transmute, marker::PhantomData};

pub enum Source<'a, O> {
    Serialized(Vec<u8>),
    Raw(O),
    Uninitialized(PhantomData<&'a O>),
    Const,
}

impl<'a, O> Source<'a, O> {
    pub fn alias(&mut self) -> Alias<'a> {
        Alias(self as *mut Source<O> as *mut (), PhantomData)
    }

    pub fn write(&mut self, o: O) -> Result<'a, ()>
    where
        O: MorphicIO<'a>,
    {
        use Source::*;
        match self {
            // Maybe this shouldnt just use bincode...
            Serialized(v) => *v = bincode::serialize(&o)?,
            Raw(v) => *v = o,
            _ => panic!("Cannot write to uninitialized or const buffer"),
        }
        Ok(())
    }

    pub fn read(self) -> Result<'a, O>
    where
        O: MorphicIO<'a>,
    {
        use Source::*;
        Ok(match self {
            Raw(v) => v,
            Serialized(v) => unsafe { bincode::deserialize(transmute(&v[..]))? },
            _ => panic!("Cannot read from uninitialized or const buffer"),
        })
    }

    pub fn uninit() -> Self {
        Self::Uninitialized(PhantomData)
    }

    fn fmt(&self) -> &'static str {
        match self {
            Source::Serialized(_) => "serialized",
            Source::Raw(_) => "raw",
            Source::Uninitialized(_) => "uninitialized",
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
        O: MorphicIO<'a>,
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

pub struct Alias<'a>(*mut (), PhantomData<&'a ()>);

impl<'a> Alias<'a> {
    pub unsafe fn attach_type<O: MorphicIO<'a>>(&self) -> &mut Source<'a, O> {
        unsafe { transmute(self.0) }
    }
}

pub const fn null() -> Source<'static, ()> {
    Source::Const
}
