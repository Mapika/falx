//! Shared helpers for the integration tests. Pulled into each test crate with
//! `mod common;`. Lives in a subdirectory so cargo does not treat it as its own
//! test target. `allow(dead_code)` because not every test uses every helper.
#![allow(dead_code)]

/// xorshift64* RNG; avoids a dev-dependency for test data generation.
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}
