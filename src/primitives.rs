use std::{any::type_name, fmt, marker::PhantomData};

use serde::{
    de::{self, SeqAccess, Visitor},
    ser::SerializeSeq,
    Deserialize, Serialize, Serializer,
};

use crate::{internal_utils::uninit, MorphicIO};

macro_rules! copy {
    ($($t: tt)*) => {
        $(unsafe impl<'a> MorphicIO<'a> for $t {
            const IS_COPY: bool = true;
        })*
    };
}

copy! (u8 u16 u32 u64 i8 i16 i32 i64 f32 f64 usize isize bool char i128 u128 ());

unsafe impl MorphicIO<'_> for String {}
unsafe impl<'a> MorphicIO<'a> for &'a str {}

unsafe impl<'a, T: MorphicIO<'a>> MorphicIO<'a> for Box<T> {}
unsafe impl<'a, T: MorphicIO<'a>> MorphicIO<'a> for Vec<T> {}
unsafe impl<'a, T: MorphicIO<'a>> MorphicIO<'a> for &'a [T] where &'a [T]: Deserialize<'a> {}

unsafe impl<'a, T: MorphicIO<'a>> MorphicIO<'a> for [T] where [T]: Deserialize<'a> {}
unsafe impl<'a, T: MorphicIO<'a>, const LEN: usize> MorphicIO<'a> for Array<T, LEN> {}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Array<T, const LEN: usize>([T; LEN]);

impl<T, const LEN: usize> std::ops::Deref for Array<T, LEN> {
    type Target = [T; LEN];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T, const LEN: usize> std::ops::DerefMut for Array<T, LEN> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T, const LEN: usize> Serialize for Array<T, LEN>
where
    T: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(self.len()))?;
        for e in &self.0 {
            seq.serialize_element(e)?;
        }
        seq.end()
    }
}

struct ArrayVisitor<T, const LEN: usize>(PhantomData<T>);

impl<'de, T, const LEN: usize> Visitor<'de> for ArrayVisitor<T, LEN>
where
    T: Deserialize<'de>,
{
    type Value = Array<T, LEN>;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str(&format!("Array<{}, {LEN}>", type_name::<T>()))
    }

    fn visit_seq<S>(self, mut seq: S) -> Result<Array<T, LEN>, S::Error>
    where
        S: SeqAccess<'de>,
    {
        let mut array: Array<T, LEN> = unsafe { uninit() };

        for n in 0..LEN {
            array[n] = seq
                .next_element()?
                .ok_or_else(|| de::Error::custom("no values in seq when looking for maximum"))?;
        }

        Ok(array)
    }
}

impl<'de, T, const LEN: usize> Deserialize<'de> for Array<T, LEN>
where
    T: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_seq(ArrayVisitor(PhantomData))
    }
}

#[test]
fn test_serde_array() {
    let mut arr = Array(['a', 'b', 'c']);

    use bincode;
    let bruh = bincode::serialize(&arr).unwrap();
    let arr2 = bincode::deserialize(&bruh).unwrap();
    assert_eq!(arr, arr2);
}
