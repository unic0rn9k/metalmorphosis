use crate::{MorphicIO, Result};
use std::{
    any::type_name,
    intrinsics::transmute,
    marker::PhantomData,
    mem::size_of,
    sync::atomic::{AtomicU16, AtomicUsize, Ordering},
};

pub enum Source<'a, O>
where
    Self: 'a,
{
    // Ready
    Serialized(Vec<u8>),
    Raw(O),
    // Pending
    Uninitialized(PhantomData<&'a O>),
    // Can never be ready
    Const,
}

// TODO: Implement SourceProvider (etc) for RcSource.
// If rc == 0 && data != uninit { not RC }
// If rc == 0 && data == UninitRc { rc++ when alias; don't allow read }
// If rc != 0 && data != uninit { rc-- when read; don't allow write or alias }
// When rc is enabled, Alias can just straight up be a future.
pub struct RcSource<'a, O> {
    data: Source<'a, O>,
    rc: AtomicU16,
}

// TODO: Some way of guaranteeing that a non-parallel SourceProvider is not used in parallel.
pub trait SourceProvider<'a, O: MorphicIO<'a>> {
    // TODO: Should return Result
    fn source(&mut self) -> Alias<'a>;
}

impl<'a, O: MorphicIO<'a>> SourceProvider<'a, O> for Source<'a, O> {
    fn source(&mut self) -> Alias<'a> {
        self.alias()
    }
}

// This is mpsc/mpmc channel, but with a static amount of slots for sending/reciving
pub struct ParallelSources {
    buffer: Vec<u8>,
    idx: AtomicUsize,
}

impl ParallelSources {
    fn new(capacity_in_bytes: usize) -> Self {
        Self {
            buffer: vec![0; capacity_in_bytes],
            idx: AtomicUsize::new(0),
        }
    }
}

impl<'a, O: MorphicIO<'a>> SourceProvider<'a, O> for ParallelSources {
    fn source(&mut self) -> Alias<'a> {
        if size_of::<O>() > self.buffer.len() {
            panic!(
                "Not enough space in ParallelSource to fit {}",
                type_name::<O>()
            );
        }
        let mut i = self.idx.fetch_add(size_of::<O>(), Ordering::SeqCst);

        if i >= self.buffer.len() {
            self.idx.store(0, Ordering::SeqCst);
            i = self.idx.fetch_add(size_of::<O>(), Ordering::SeqCst);
        }

        if i >= self.buffer.len() {
            panic!("ParallelSources failed to allocate!")
        }

        // source should be able to have different types, fx RcSource.
        let source = unsafe { std::mem::transmute::<_, &mut Source<'a, O>>(&mut self.buffer[i]) };
        *source = Source::uninit();
        source.alias()
    }
}

impl<'a, O> Source<'a, O> {
    pub fn clear(&mut self) {
        *self = Self::Uninitialized(PhantomData);
    }
    pub fn alias(&mut self) -> Alias<'a> {
        Alias(self as *mut Source<O> as *mut (), PhantomData)
    }

    // TODO: Notify parent that data has been written. This should props be done in in the task handle, tho.
    pub fn write(&mut self, o: O) -> Result<'a, ()>
    where
        O: MorphicIO<'a>,
    {
        use Source::*;
        match self {
            // Maybe this shouldn't just use bincode...
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

// TODO: Mby replace `*mut ()` with `&'b Source<'a, ()>`
#[derive(Clone, Copy)]
pub struct Alias<'a>(*mut (), PhantomData<&'a ()>);

impl<'a> Alias<'a> {
    pub unsafe fn attach_type<O: MorphicIO<'a>>(&self) -> &mut Source<'a, O> {
        unsafe { transmute(self.0) }
    }
}

pub const NULL: Source<'static, ()> = Source::Const;

#[macro_export]
macro_rules! null_alias {
    () => {{
        //#[allow(const_item_mutation)]
        //buffer::NULL.alias()
        todo!()
    }};
}

pub const RAW: char = 'r';
pub const SERIALIZED: char = 's';
