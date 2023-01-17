use std::{
    any::{type_name, Any},
    cell::UnsafeCell,
    mem::transmute,
};

use serde::{Deserialize, Serialize};

use crate::error::*;

// TODO: Interior mutability in Buffer, instead of Node
pub struct Buffer {
    data: UnsafeCell<Box<dyn Any>>,
    de: fn(&[u8], *mut ()) -> Result<()>,
    se: fn(*const ()) -> Result<Vec<u8>>,
}

impl Buffer {
    pub fn new<T: Serialize + Deserialize<'static> + Sync + 'static>() -> Self {
        unsafe {
            Buffer {
                data: UnsafeCell::new(Box::<T>::new_uninit().assume_init()),
                de: |b, out| {
                    *(out as *mut T) = bincode::deserialize(transmute(b))?;
                    Ok(())
                },
                se: |v| Ok(bincode::serialize::<T>(&*(v as *const T))?),
            }
        }
    }

    pub unsafe fn serialize(&self) -> Vec<u8> {
        (self.se)(self.ptr()).expect("Buffer serialization failed")
    }

    pub unsafe fn deserialize(&self, data: &[u8]) {
        (self.de)(data, self.mut_ptr()).expect("Buffer deserialization failed")
    }

    pub unsafe fn ptr(&self) -> *const () {
        transmute::<&dyn Any, (*const (), &())>(&*self.data.get()).0
    }
    pub unsafe fn mut_ptr(&self) -> *mut () {
        transmute::<&mut dyn Any, (*mut (), &())>(&mut *self.data.get()).0
    }

    pub unsafe fn downcast_ptr_mut<T: 'static>(&self) -> *mut T {
        unsafe {
            (*self.data.get()).downcast_mut().unwrap_or_else(|| {
                panic!(
                    "Tried to get output with incorrect runtime type. Expected {}",
                    type_name::<T>()
                )
            })
        }
    }

    pub unsafe fn downcast_ptr<T: 'static>(&self) -> *const T {
        unsafe {
            (*self.data.get()).downcast_mut().unwrap_or_else(|| {
                panic!(
                    "Tried to get output with incorrect runtime type. Expected {}",
                    type_name::<T>()
                )
            })
        }
    }
}
