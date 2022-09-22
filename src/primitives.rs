use crate::MorphicIO;

macro_rules! def {
    ($($t: ty)*) => {
        $(unsafe impl MorphicIO for $t {
            const IS_COPY: bool = true;
        })*
    };
}

def! {u8 u32 u64 i8 i32 i64 f32 f64 usize isize bool}
