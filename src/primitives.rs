use crate::MorphicIO;

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
