pub unsafe fn uninit<T>() -> T {
    std::mem::MaybeUninit::uninit().assume_init()
}
