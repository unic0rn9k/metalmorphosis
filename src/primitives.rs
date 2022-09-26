use crate::MorphicIO;

macro_rules! copy {
    ($($t: tt)*) => {
        $(unsafe impl MorphicIO for $t {
            const IS_COPY: bool = true;
        })*
    };
}

copy! (u8 u32 u64 i8 i32 i64 f32 f64 usize isize bool char i128 u128 ());

unsafe impl MorphicIO for String {}
unsafe impl<'a> MorphicIO for &'a str {}

unsafe impl<T: MorphicIO> MorphicIO for Box<T> {}
unsafe impl<T: MorphicIO> MorphicIO for Vec<T> {}
