//! The typed-AST emitter renders one tree to multiple backends. These tests
//! pin the exact Rust and CUDA-C output for a representative kernel helper, and
//! check the C backend rejects constructs it cannot faithfully express.

use falx::emit::ast::*;
use falx::emit::{emit_c, emit_rust};

/// `fn checksum(seed: u64, bytes: &[u8]) -> u64` — a rolling FNV-1a fold.
/// Exercises: a doc comment, an inline attribute, a slice parameter, a body
/// comment, an indexed loop over the slice length, a cast, a compound
/// assignment, and wrapping multiplication.
fn checksum_program() -> Vec<Item> {
    let f = Func::new("checksum")
        .public()
        .attr("inline")
        .doc("Fold `bytes` into a rolling FNV-1a checksum.")
        .param("seed", Type::name("u64"))
        .param("bytes", Type::slice(Type::name("u8")))
        .ret(Type::name("u64"))
        .body(vec![
            Stmt::let_("h", true, Type::name("u64"), Expr::path("seed")),
            Stmt::line_comment("one wrapping multiply-xor per byte"),
            Stmt::for_range(
                "i",
                Expr::int(0),
                Expr::slice_len(Expr::path("bytes")),
                vec![
                    Stmt::assign_op(
                        Expr::path("h"),
                        BinOp::BitXor,
                        Expr::cast(
                            Expr::index(Expr::path("bytes"), Expr::path("i")),
                            Type::name("u64"),
                        ),
                    ),
                    Stmt::assign(
                        Expr::path("h"),
                        Expr::binary(BinOp::WrapMul, Expr::path("h"), Expr::hex(0x100000001b3)),
                    ),
                ],
            ),
            Stmt::ret(Expr::path("h")),
        ]);
    vec![Item::from(f)]
}

const EXPECTED_RUST: &str = r#"/// Fold `bytes` into a rolling FNV-1a checksum.
#[inline]
pub fn checksum(seed: u64, bytes: &[u8]) -> u64 {
    let mut h: u64 = seed;
    // one wrapping multiply-xor per byte
    for i in 0..bytes.len() {
        h ^= bytes[i] as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    return h;
}
"#;

const EXPECTED_C: &str = r#"// Fold `bytes` into a rolling FNV-1a checksum.
inline uint64_t checksum(uint64_t seed, const uint8_t* bytes, size_t bytes_len) {
    uint64_t h = seed;
    // one wrapping multiply-xor per byte
    for (size_t i = 0; i < bytes_len; i++) {
        h ^= (uint64_t)bytes[i];
        h = h * 0x100000001b3ULL;
    }
    return h;
}
"#;

#[test]
fn renders_idiomatic_rust() {
    assert_eq!(emit_rust(&checksum_program()).unwrap(), EXPECTED_RUST);
}

#[test]
fn renders_idiomatic_c() {
    assert_eq!(emit_c(&checksum_program()).unwrap(), EXPECTED_C);
}

#[test]
fn comments_survive_to_both_backends() {
    let rust = emit_rust(&checksum_program()).unwrap();
    let c = emit_c(&checksum_program()).unwrap();
    assert!(rust.contains("/// Fold `bytes` into a rolling FNV-1a checksum."));
    assert!(c.contains("// Fold `bytes` into a rolling FNV-1a checksum."));
    assert!(rust.contains("// one wrapping multiply-xor per byte"));
    assert!(c.contains("// one wrapping multiply-xor per byte"));
}

#[test]
fn precedence_parenthesizes_only_where_needed() {
    // (a + b) * c  must keep the parens; a + b * c must not.
    let needs = Expr::binary(
        BinOp::Mul,
        Expr::binary(BinOp::Add, Expr::path("a"), Expr::path("b")),
        Expr::path("c"),
    );
    let flat = Expr::binary(
        BinOp::Add,
        Expr::path("a"),
        Expr::binary(BinOp::Mul, Expr::path("b"), Expr::path("c")),
    );
    let prog = |e: Expr| {
        vec![Item::from(
            Func::new("f")
                .ret(Type::name("u64"))
                .body(vec![Stmt::ret(e)]),
        )]
    };
    assert!(
        emit_rust(&prog(needs.clone()))
            .unwrap()
            .contains("return (a + b) * c;")
    );
    assert!(
        emit_rust(&prog(flat.clone()))
            .unwrap()
            .contains("return a + b * c;")
    );
    assert!(
        emit_c(&prog(needs))
            .unwrap()
            .contains("return (a + b) * c;")
    );
    assert!(emit_c(&prog(flat)).unwrap().contains("return a + b * c;"));
}

#[test]
fn c_backend_rejects_constructs_it_cannot_express() {
    // A slice length only makes sense for a slice *parameter* in C.
    let bad_len = vec![Item::from(
        Func::new("f")
            .ret(Type::name("u64"))
            .body(vec![Stmt::ret(Expr::slice_len(Expr::int(3)))]),
    )];
    assert!(emit_c(&bad_len).is_err());

    // C has no type inference: a `let` without a type cannot be rendered...
    let untyped_let = vec![Item::from(Func::new("g").body(vec![Stmt::Let {
        name: "x".into(),
        mutable: false,
        ty: None,
        init: Expr::int(1),
    }]))];
    assert!(emit_c(&untyped_let).is_err());
    // ...but Rust infers it just fine.
    assert!(emit_rust(&untyped_let).is_ok());
}
