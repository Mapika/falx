#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let mut a = Vec::new();
    falx::kernels::csv::index_structurals(data, &mut a);
    let mut b = Vec::new();
    falx::scalar::index_structurals(data, &mut b);
    assert_eq!(a, b);
});
