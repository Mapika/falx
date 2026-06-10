#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let mut a = Vec::new();
    vexel::kernels::csv::index_structurals(data, &mut a);
    let mut b = Vec::new();
    vexel::scalar::index_structurals(data, &mut b);
    assert_eq!(a, b);
});
