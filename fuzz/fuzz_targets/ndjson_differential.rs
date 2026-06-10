#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let mut a = Vec::new();
    falx::kernels::ndjson::index_structurals(data, &mut a);
    let mut b = Vec::new();
    falx::scalar::index_structurals_spec(data, &[b'\n'], Some(b'"'), Some(b'\\'), &mut b);
    assert_eq!(a, b);
});
