//! Lowering: `ir::Graph` → [`emit::ast`](super::ast) → a structural indexer.
//! **Milestones 1–2 toward `codegen` parity.**
//!
//! It produces a *blockwise* kernel: a direct, unrolled specialization of
//! [`crate::interp`] — the same 64-byte-block / per-node-`u64` / threaded-carry
//! structure the real SIMD kernels use. The whole `u64` bit-algebra (and/or/
//! not/xor, the carries, `prefix_xor`, the ripple-`Add`) is *already* exactly
//! what a SIMD kernel computes; the only ISA-specific seam is `class_mask`
//! (byte classification). M2 makes that seam AVX2 (`cmpeq` + `movemask`) with a
//! scalar / non-x86 fallback, so the emitted kernel is genuinely SIMD-
//! accelerated while staying portable and correct.
//!
//! Coverage: every `Op` in the IR. `PrefixXor` lowers to typed AST; `Class`
//! lowers to a `class_mask` call (the AVX2/scalar helper, emitted via the `Raw`
//! escape hatch because it touches intrinsics); the three-state `Regions`
//! resolver is also `Raw` (mirroring `interp::resolve_regions`). The block
//! structure and the rest of the algebra are fully typed. The SIMD classify
//! covers x86 (AVX-512 → AVX2 → scalar) and aarch64 (NEON), with PCLMULQDQ
//! quote-parity on x86 — matching production's approach. Beyond the index,
//! `lower_parser` (M3) and `lower_columns` (M4) add records/fields and typed
//! columns, `lower_index_c` (M5) lowers the index to CUDA-C, `nested_function`
//! (M6) a JSON bracket tape, and `count_structural_function` (M7) stat sinks —
//! one IR graph, multiple target languages.

use super::ast::{BinOp, Block, Expr, Func, Item, Stmt, Type, UnOp};
use crate::formats::{self, Dialect};
use crate::ir::{Graph, Op};

/// Lower a dialect's structural-indexing graph to a self-contained module: the
/// helpers it needs plus a `pub fn index_structurals(data, out)`.
pub fn lower_indexer(dialect: &Dialect) -> Vec<Item> {
    let graph = formats::delimited(dialect);
    let refs = [&graph];
    let mut items = needed_helpers(&refs);
    items.extend(rust_index_items(&graph, "index_structurals"));
    items
}

/// Lower a dialect to a self-contained module: the indexer (see
/// [`lower_indexer`]) plus a `parse` that splits records and fields over the
/// structural index. **Milestone 3.**
pub fn lower_parser(dialect: &Dialect) -> Vec<Item> {
    let mut items = lower_indexer(dialect);
    items.push(parse_function("index_structurals"));
    items
}

/// The helper items required by `graphs` (their union): `class_mask` and
/// `push_indexes` (the output scatter) always, `prefix_xor` if any graph uses
/// [`Op::PrefixXor`], the region resolver if any uses [`Op::Regions`].
pub fn needed_helpers(graphs: &[&Graph]) -> Vec<Item> {
    let uses = |pred: fn(&Op) -> bool| graphs.iter().any(|g| g.nodes().iter().any(pred));
    let mut items = vec![class_mask_helper(), push_indexes_helper()];
    if uses(|o| matches!(o, Op::PrefixXor(_))) {
        items.push(prefix_xor_helper());
    }
    if uses(|o| matches!(o, Op::Regions(..))) {
        items.push(region_helpers());
    }
    // The baked aarch64 NEON variant (emitted for every non-Regions graph) needs
    // the NEON classify; its PMULL prefix-xor only when some graph has quotes.
    let any_baked = graphs
        .iter()
        .any(|g| !g.nodes().iter().any(|o| matches!(o, Op::Regions(..))));
    if any_baked {
        items.push(neon_ops_mod(uses(|o| matches!(o, Op::PrefixXor(_)))));
    }
    items
}

/// The aarch64 NEON helpers used by the baked NEON whole-loop variant: a
/// `movemask`-equivalent fold + `eq_mask` classify, plus a PMULL `prefix_xor`
/// when `has_px`. Mirrors the production kernel's NEON path; NEON is baseline on
/// aarch64 so only PMULL needs a `target_feature`.
fn neon_ops_mod(has_px: bool) -> Item {
    let prefix_xor = if has_px {
        r#"
    /// Inclusive prefix-XOR via PMULL (carryless multiply by all-ones) — the
    /// aarch64 stand-in for x86 PCLMULQDQ.
    #[target_feature(enable = "aes")]
    pub unsafe fn prefix_xor(mask: u64) -> u64 {
        vmull_p64(mask, !0u64) as u64
    }
"#
    } else {
        ""
    };
    Item::Raw(format!(
        r#"/// NEON classify (+ PMULL prefix-xor) for the baked aarch64 whole-loop variant.
#[cfg(target_arch = "aarch64")]
mod neon_ops {{
    use core::arch::aarch64::*;

    /// NEON has no movemask: AND each lane with its bit value, then fold the four
    /// 16-byte compares to one dense `u64` (bit j = byte j) with pairwise adds —
    /// bit-identical to x86 movemask.
    fn movemask(c0: uint8x16_t, c1: uint8x16_t, c2: uint8x16_t, c3: uint8x16_t) -> u64 {{
        unsafe {{
            let bits = vreinterpretq_u8_u64(vdupq_n_u64(0x8040_2010_0804_0201));
            let s0 = vpaddq_u8(vandq_u8(c0, bits), vandq_u8(c1, bits));
            let s1 = vpaddq_u8(vandq_u8(c2, bits), vandq_u8(c3, bits));
            let s2 = vpaddq_u8(s0, s1);
            let s3 = vpaddq_u8(s2, s2);
            vgetq_lane_u64::<0>(vreinterpretq_u64_u8(s3))
        }}
    }}

    /// 64-bit class mask for `byte` over four preloaded 16-byte lanes.
    pub fn eq_mask(b0: uint8x16_t, b1: uint8x16_t, b2: uint8x16_t, b3: uint8x16_t, byte: u8) -> u64 {{
        unsafe {{
            let needle = vdupq_n_u8(byte);
            movemask(
                vceqq_u8(b0, needle),
                vceqq_u8(b1, needle),
                vceqq_u8(b2, needle),
                vceqq_u8(b3, needle),
            )
        }}
    }}
{prefix_xor}}}"#
    ))
}

/// `push_indexes(mask, base, out)` — the output scatter: reserve capacity once,
/// then write set-bit positions in unconditional chunks of 8 via a raw pointer
/// (overshoot ≤7 masked by a single `set_len`), avoiding a per-bit `Vec::push`
/// capacity check. Matches the production kernel's hot extraction path.
fn push_indexes_helper() -> Item {
    Item::Raw(
        r#"/// Scatter the set bits of `mask` to `out` as `base + bit`. Reserves once and
/// writes unconditional chunks of 8 via raw pointer (overshoot masked by
/// `set_len`), so there is no per-bit `push` capacity check.
#[inline]
fn push_indexes(mut mask: u64, base: u32, out: &mut Vec<u32>) {
    let count = mask.count_ones() as usize;
    if count == 0 {
        return;
    }
    let len = out.len();
    out.reserve(count + 8);
    // SAFETY: capacity covers len + count + 8; chunked writes overshoot by at
    // most 7 entries and set_len exposes only the real ones.
    unsafe {
        let mut ptr = out.as_mut_ptr().add(len);
        let mut remaining = count as isize;
        while remaining > 0 {
            let mut j = 0;
            while j < 8 {
                *ptr.add(j) = base + mask.trailing_zeros();
                mask &= mask.wrapping_sub(1);
                j += 1;
            }
            ptr = ptr.add(8);
            remaining -= 8;
        }
        out.set_len(len + count);
    }
}"#
        .into(),
    )
}

/// Lower one graph to a single `pub fn <name>(data: &[u8], out: &mut Vec<u32>)`.
/// (Helpers are emitted separately via [`needed_helpers`] so several graphs can
/// share one set.)
pub fn index_function(graph: &Graph, name: &str) -> Item {
    emit_index(Target::Rust, graph, name)
}

/// Lower a graph's structural index for `target`: the shared blockwise structure
/// (per-node carries, the 64-byte loop, the zero-padded tail); the few
/// language-specific leaves come from the [`Target`].
fn emit_index(target: Target, graph: &Graph, name: &str) -> Item {
    let live = live_nodes(graph);
    let out_id = graph.output().0 as usize;

    let mut body: Vec<Stmt> = Vec::new();
    // Carry state for every live stateful node (seeded shifts start at 1).
    for (i, op) in graph.nodes().iter().enumerate() {
        if live[i] && is_stateful(op) {
            let init = u64::from(matches!(op, Op::ShiftLeft1Seeded(_)));
            body.push(Stmt::let_(
                format!("carry_{i}"),
                true,
                Type::name("u64"),
                Expr::int(init),
            ));
        }
    }
    body.push(Stmt::let_(
        "offset",
        true,
        Type::name("usize"),
        Expr::int(0),
    ));

    // Full 64-byte blocks.
    let mut loop_body = block_body(target, graph, &live, out_id, false);
    loop_body.push(Stmt::assign_op(
        Expr::path("offset"),
        BinOp::Add,
        Expr::int(64),
    ));
    body.push(Stmt::While {
        cond: Expr::binary(
            BinOp::Le,
            Expr::binary(BinOp::Add, Expr::path("offset"), Expr::int(64)),
            target.len(),
        ),
        body: Block(loop_body),
    });

    // Zero-padded tail block.
    body.push(Stmt::let_(
        "rem",
        false,
        Type::name("usize"),
        Expr::binary(BinOp::Sub, target.len(), Expr::path("offset")),
    ));
    body.push(Stmt::If {
        cond: Expr::binary(BinOp::Gt, Expr::path("rem"), Expr::int(0)),
        then: Block(block_body(target, graph, &live, out_id, true)),
        els: None,
    });

    Item::from(target.index_func(name, body))
}

/// `pub fn parse(data) -> Vec<Vec<&[u8]>>` — records (split on unquoted `\n`,
/// CRLF trimmed) of raw field spans (split on the other structural bytes).
/// Dialect-independent: it just walks the structural index that `index_name`
/// produces, classifying each position as a record terminator (`\n`) or a field
/// separator. The field/record byte spans match the production kernel's
/// `records()` / `field_raw`.
pub fn parse_function(index_name: &str) -> Item {
    // The CRLF-trim guard: `end > record_start && data[end - 1] == b'\r'`.
    let crlf_guard = || {
        Expr::binary(
            BinOp::AndAnd,
            Expr::binary(BinOp::Gt, Expr::path("end"), Expr::path("record_start")),
            Expr::binary(
                BinOp::Eq,
                Expr::index(
                    Expr::path("data"),
                    Expr::binary(BinOp::Sub, Expr::path("end"), Expr::int(1)),
                ),
                Expr::raw("b'\\r'"),
            ),
        )
    };
    let trim_crlf = || Stmt::If {
        cond: crlf_guard(),
        then: Block(vec![Stmt::assign_op(
            Expr::path("end"),
            BinOp::Sub,
            Expr::int(1),
        )]),
        els: None,
    };
    let push_field = |slice: &str| {
        Stmt::Expr(Expr::call(
            Expr::path("fields.push"),
            vec![Expr::raw(slice)],
        ))
    };
    let advance = |var: &str| {
        Stmt::assign(
            Expr::path(var),
            Expr::binary(BinOp::Add, Expr::path("p"), Expr::int(1)),
        )
    };

    let body = vec![
        Stmt::let_(
            "idx",
            true,
            Type::Raw("Vec<u32>".into()),
            Expr::raw("Vec::new()"),
        ),
        Stmt::Expr(Expr::call(
            Expr::path(index_name),
            vec![Expr::path("data"), Expr::raw("&mut idx")],
        )),
        Stmt::let_(
            "records",
            true,
            Type::Raw("Vec<Vec<&[u8]>>".into()),
            Expr::raw("Vec::new()"),
        ),
        Stmt::let_(
            "fields",
            true,
            Type::Raw("Vec<&[u8]>".into()),
            Expr::raw("Vec::new()"),
        ),
        Stmt::let_("field_start", true, Type::name("usize"), Expr::int(0)),
        Stmt::let_("record_start", true, Type::name("usize"), Expr::int(0)),
        Stmt::ForRange {
            var: "k".into(),
            start: Expr::int(0),
            end: Expr::call(Expr::path("idx.len"), vec![]),
            body: Block(vec![
                Stmt::Let {
                    name: "p".into(),
                    mutable: false,
                    ty: None,
                    init: Expr::cast(
                        Expr::index(Expr::path("idx"), Expr::path("k")),
                        Type::name("usize"),
                    ),
                },
                Stmt::If {
                    cond: Expr::binary(
                        BinOp::Eq,
                        Expr::index(Expr::path("data"), Expr::path("p")),
                        Expr::raw("b'\\n'"),
                    ),
                    then: Block(vec![
                        Stmt::let_("end", true, Type::name("usize"), Expr::path("p")),
                        trim_crlf(),
                        push_field("&data[field_start..end]"),
                        Stmt::Expr(Expr::call(
                            Expr::path("records.push"),
                            vec![Expr::raw("std::mem::take(&mut fields)")],
                        )),
                        advance("field_start"),
                        advance("record_start"),
                    ]),
                    els: Some(Block(vec![
                        push_field("&data[field_start..p]"),
                        advance("field_start"),
                    ])),
                },
            ]),
        },
        // Trailing record with no final newline.
        Stmt::If {
            cond: Expr::binary(
                BinOp::Lt,
                Expr::path("record_start"),
                Expr::call(Expr::path("data.len"), vec![]),
            ),
            then: Block(vec![
                Stmt::let_(
                    "end",
                    true,
                    Type::name("usize"),
                    Expr::call(Expr::path("data.len"), vec![]),
                ),
                trim_crlf(),
                push_field("&data[field_start..end]"),
                Stmt::Expr(Expr::call(
                    Expr::path("records.push"),
                    vec![Expr::path("fields")],
                )),
            ]),
            els: None,
        },
        Stmt::ret(Expr::path("records")),
    ];

    Item::from(
        Func::new("parse")
            .public()
            .doc("Split `data` into records of raw field spans, walking the")
            .doc("structural index. Byte-equivalent to the production kernel's")
            .doc("`records()` / `field_raw`. Generated by `emit::lower`.")
            .param("data", Type::slice(Type::name("u8")))
            .ret(Type::Raw("Vec<Vec<&[u8]>>".into()))
            .body(body),
    )
}

/// Lower a dialect to a module with the parser plus a typed column projection
/// (`column_i64`). **Milestone 4.**
pub fn lower_columns(dialect: &Dialect) -> Vec<Item> {
    let mut items = lower_parser(dialect);
    items.push(parse_i64_helper());
    items.push(column_i64_function());
    items
}

/// `fn parse_i64(field) -> Option<i64>` — the typed-projection primitive
/// (`None` on invalid UTF-8 or a malformed integer). Emitted via `Raw`.
pub fn parse_i64_helper() -> Item {
    Item::Raw(
        r#"/// Parse a field's bytes as an i64 (None on invalid UTF-8 or bad integer).
fn parse_i64(field: &[u8]) -> Option<i64> {
    std::str::from_utf8(field).ok().and_then(|s| s.parse::<i64>().ok())
}"#
        .into(),
    )
}

/// `pub fn column_i64(data, col) -> Vec<Option<i64>>` — project field `col` of
/// every record and parse it as an i64 (`None` when the record has no such
/// field). Built on `parse` + `parse_i64`.
pub fn column_i64_function() -> Item {
    let body = vec![
        Stmt::Let {
            name: "recs".into(),
            mutable: false,
            ty: None,
            init: Expr::call(Expr::path("parse"), vec![Expr::path("data")]),
        },
        Stmt::let_(
            "out",
            true,
            Type::Raw("Vec<Option<i64>>".into()),
            Expr::raw("Vec::new()"),
        ),
        Stmt::ForRange {
            var: "r".into(),
            start: Expr::int(0),
            end: Expr::call(Expr::path("recs.len"), vec![]),
            body: Block(vec![Stmt::If {
                cond: Expr::binary(BinOp::Lt, Expr::path("col"), Expr::raw("recs[r].len()")),
                then: Block(vec![Stmt::Expr(Expr::call(
                    Expr::path("out.push"),
                    vec![Expr::call(
                        Expr::path("parse_i64"),
                        vec![Expr::raw("recs[r][col]")],
                    )],
                ))]),
                els: Some(Block(vec![Stmt::Expr(Expr::call(
                    Expr::path("out.push"),
                    vec![Expr::raw("None")],
                ))])),
            }]),
        },
        Stmt::ret(Expr::path("out")),
    ];
    Item::from(
        Func::new("column_i64")
            .public()
            .doc("Project field `col` of every record, parsed as i64 (None if the")
            .doc("record has no such field). Generated by `emit::lower`.")
            .param("data", Type::slice(Type::name("u8")))
            .param("col", Type::name("usize"))
            .ret(Type::Raw("Vec<Option<i64>>".into()))
            .body(body),
    )
}

/// Lower a dialect to a module with the indexer plus `parse_nested`. **M6.**
pub fn lower_nested(dialect: &Dialect) -> Vec<Item> {
    let mut items = lower_indexer(dialect);
    items.push(nested_function("index_structurals"));
    items
}

/// `pub fn parse_nested(data) -> Vec<(usize, usize)>` — matched bracket pairs
/// `(open, close)` for a nesting dialect (JSON), found by walking the
/// structural index with a stack. Brackets inside strings never appear in the
/// index, so they are correctly ignored. M6: the nested-tape surface.
pub fn nested_function(index_name: &str) -> Item {
    let is_open = Expr::binary(
        BinOp::OrOr,
        Expr::binary(BinOp::Eq, Expr::path("b"), Expr::raw("b'{'")),
        Expr::binary(BinOp::Eq, Expr::path("b"), Expr::raw("b'['")),
    );
    let is_close = Expr::binary(
        BinOp::OrOr,
        Expr::binary(BinOp::Eq, Expr::path("b"), Expr::raw("b'}'")),
        Expr::binary(BinOp::Eq, Expr::path("b"), Expr::raw("b']'")),
    );
    let pop_and_pair = Stmt::If {
        cond: Expr::unary(UnOp::Not, Expr::call(Expr::path("stack.is_empty"), vec![])),
        then: Block(vec![
            Stmt::Let {
                name: "open".into(),
                mutable: false,
                ty: None,
                init: Expr::raw("stack.pop().unwrap()"),
            },
            Stmt::Expr(Expr::call(
                Expr::path("pairs.push"),
                vec![Expr::raw("(open, p)")],
            )),
        ]),
        els: None,
    };
    let body = vec![
        Stmt::let_(
            "idx",
            true,
            Type::Raw("Vec<u32>".into()),
            Expr::raw("Vec::new()"),
        ),
        Stmt::Expr(Expr::call(
            Expr::path(index_name),
            vec![Expr::path("data"), Expr::raw("&mut idx")],
        )),
        Stmt::let_(
            "pairs",
            true,
            Type::Raw("Vec<(usize, usize)>".into()),
            Expr::raw("Vec::new()"),
        ),
        Stmt::let_(
            "stack",
            true,
            Type::Raw("Vec<usize>".into()),
            Expr::raw("Vec::new()"),
        ),
        Stmt::ForRange {
            var: "k".into(),
            start: Expr::int(0),
            end: Expr::call(Expr::path("idx.len"), vec![]),
            body: Block(vec![
                Stmt::Let {
                    name: "p".into(),
                    mutable: false,
                    ty: None,
                    init: Expr::cast(
                        Expr::index(Expr::path("idx"), Expr::path("k")),
                        Type::name("usize"),
                    ),
                },
                Stmt::Let {
                    name: "b".into(),
                    mutable: false,
                    ty: None,
                    init: Expr::index(Expr::path("data"), Expr::path("p")),
                },
                Stmt::If {
                    cond: is_open,
                    then: Block(vec![Stmt::Expr(Expr::call(
                        Expr::path("stack.push"),
                        vec![Expr::path("p")],
                    ))]),
                    els: Some(Block(vec![Stmt::If {
                        cond: is_close,
                        then: Block(vec![pop_and_pair]),
                        els: None,
                    }])),
                },
            ]),
        },
        Stmt::ret(Expr::path("pairs")),
    ];
    Item::from(
        Func::new("parse_nested")
            .public()
            .doc("Matched bracket pairs (open, close) over the structural index.")
            .doc("Generated by `emit::lower`.")
            .param("data", Type::slice(Type::name("u8")))
            .ret(Type::Raw("Vec<(usize, usize)>".into()))
            .body(body),
    )
}

/// `pub fn <name>(data) -> usize` — count structural bytes equal to `needle`
/// (a byte literal like `b'\n'`), divided by `divisor`. A stat-sink primitive:
/// `fastq_read_count` is newline-count / 4; `logfmt_pair_count` is `=`-count.
/// **M7.**
pub fn count_structural_function(
    index_name: &str,
    needle: &str,
    divisor: usize,
    name: &str,
) -> Item {
    let ret = if divisor == 1 {
        Stmt::ret(Expr::path("count"))
    } else {
        Stmt::ret(Expr::raw(format!("count / {divisor}")))
    };
    let body = vec![
        Stmt::let_(
            "idx",
            true,
            Type::Raw("Vec<u32>".into()),
            Expr::raw("Vec::new()"),
        ),
        Stmt::Expr(Expr::call(
            Expr::path(index_name),
            vec![Expr::path("data"), Expr::raw("&mut idx")],
        )),
        Stmt::let_("count", true, Type::name("usize"), Expr::int(0)),
        Stmt::ForRange {
            var: "k".into(),
            start: Expr::int(0),
            end: Expr::call(Expr::path("idx.len"), vec![]),
            body: Block(vec![Stmt::If {
                cond: Expr::binary(
                    BinOp::Eq,
                    Expr::index(
                        Expr::path("data"),
                        Expr::cast(
                            Expr::index(Expr::path("idx"), Expr::path("k")),
                            Type::name("usize"),
                        ),
                    ),
                    Expr::raw(needle),
                ),
                then: Block(vec![Stmt::assign_op(
                    Expr::path("count"),
                    BinOp::Add,
                    Expr::int(1),
                )]),
                els: None,
            }]),
        },
        ret,
    ];
    Item::from(
        Func::new(name)
            .public()
            .doc("Stat sink: a count over the structural index. Generated by")
            .doc("`emit::lower`.")
            .param("data", Type::slice(Type::name("u8")))
            .ret(Type::name("usize"))
            .body(body),
    )
}

// ---- Milestone 5: a CUDA-C backend lowering of the structural index. -------
//
// One lowering, a second language. The block structure and `u64` algebra render
// straight through `emit_c`; the C-specific leaves (output-pointer writes, the
// `class_mask` compound-literal call, `__builtin_ctzll`, the tail copy) use the
// `Raw` escape hatch, and every `let` is typed (C has no inference). Regions
// dialects (csv_hash) are not yet ported to C.

/// Lower a dialect's structural index to CUDA-C: `void index_structurals(const
/// uint8_t* data, size_t len, uint32_t* out, size_t* out_count)` + helpers.
/// Render with [`emit_c`](super::emit_c).
pub fn lower_index_c(dialect: &Dialect) -> Vec<Item> {
    let graph = formats::delimited(dialect);
    let mut items = needed_c_helpers(&graph);
    items.push(emit_index(Target::C, &graph, "index_structurals"));
    items
}

fn needed_c_helpers(graph: &Graph) -> Vec<Item> {
    let mut items = vec![class_mask_c_helper()];
    if graph.nodes().iter().any(|o| matches!(o, Op::PrefixXor(_))) {
        items.push(prefix_xor_c_helper());
    }
    if graph.nodes().iter().any(|o| matches!(o, Op::Regions(..))) {
        items.push(region_c_helper());
    }
    items
}

fn class_mask_c_helper() -> Item {
    Item::Raw(
        r#"static uint64_t class_mask(const uint8_t* block, const uint8_t* members, size_t nmembers) {
    uint64_t m = 0;
    for (size_t i = 0; i < 64; i++) {
        for (size_t j = 0; j < nmembers; j++) {
            if (block[i] == members[j]) { m |= (uint64_t)1 << i; }
        }
    }
    return m;
}"#
        .into(),
    )
}

fn prefix_xor_c_helper() -> Item {
    Item::Raw(
        r#"static uint64_t prefix_xor(uint64_t x) {
    x ^= x << 1;
    x ^= x << 2;
    x ^= x << 4;
    x ^= x << 8;
    x ^= x << 16;
    x ^= x << 32;
    return x;
}"#
        .into(),
    )
}

fn region_c_helper() -> Item {
    Item::Raw(
        r#"static uint64_t range_mask(uint32_t from, uint32_t to) {
    uint64_t hi = (to >= 64) ? ~(uint64_t)0 : (((uint64_t)1 << to) - 1);
    return hi & ~(((uint64_t)1 << from) - 1);
}

/// Sequential three-state region resolver (normal/quote/comment); `state`
/// carries the region state across blocks. Mirrors `interp::resolve_regions`.
static uint64_t resolve_regions(uint64_t q, uint64_t s, uint64_t n, uint64_t* state) {
    uint64_t inert = 0;
    uint32_t run_start = 0;
    uint64_t events = q | s | n;
    while (events != 0) {
        uint32_t p = (uint32_t)__builtin_ctzll(events);
        uint64_t bit = (uint64_t)1 << p;
        if (*state == 1) {
            if (q & bit) { inert |= range_mask(run_start, p); *state = 0; }
        } else if (*state == 2) {
            if (n & bit) { inert |= range_mask(run_start, p); *state = 0; }
        } else {
            if (q & bit) { *state = 1; run_start = p; }
            else if (s & bit) { *state = 2; run_start = p; }
        }
        events &= events - 1;
    }
    if (*state != 0) { inert |= range_mask(run_start, 64); }
    return inert;
}"#
        .into(),
    )
}

/// Target language for the per-op lowering. The block structure and `u64`
/// bit-algebra are shared; a `Target` supplies only the few leaves that differ
/// between the Rust and CUDA-C backends.
#[derive(Clone, Copy)]
enum Target {
    Rust,
    C,
}

impl Target {
    /// Type for a derived node binding — Rust infers it, C must spell `u64`.
    fn node_ty(self) -> Option<Type> {
        match self {
            Target::Rust => None,
            Target::C => Some(Type::name("u64")),
        }
    }

    /// `class_mask(…)` call for a class with these member bytes.
    fn class_expr(self, members: &[u8]) -> Expr {
        let list = members
            .iter()
            .map(|b| b.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        match self {
            Target::Rust => Expr::call(
                Expr::path("class_mask"),
                vec![Expr::path("block"), Expr::raw(format!("&[{list}]"))],
            ),
            Target::C => Expr::raw(format!(
                "class_mask(block, (const uint8_t[]){{{list}}}, {})",
                members.len()
            )),
        }
    }

    /// Wrapping add: Rust `a.wrapping_add(b)`, C plain `a + b` (unsigned wraps).
    fn add_op(self) -> BinOp {
        match self {
            Target::Rust => BinOp::WrapAdd,
            Target::C => BinOp::Add,
        }
    }

    /// Carry reference for `resolve_regions`: Rust `&mut x`, C `&x`.
    fn carry_ref(self, carry: &str) -> Expr {
        match self {
            Target::Rust => Expr::raw(format!("&mut {carry}")),
            Target::C => Expr::raw(format!("&{carry}")),
        }
    }

    /// Bind `block` to one 64-byte window, with any tail prelude before it.
    fn block_bind(self, tail: bool) -> Vec<Stmt> {
        let slice = || Type::slice(Type::name("u8"));
        let ptr = || Type::Ref(Box::new(Type::name("u8")));
        match (self, tail) {
            (Target::Rust, false) => vec![Stmt::let_(
                "block",
                false,
                slice(),
                Expr::raw("&data[offset..offset + 64]"),
            )],
            (Target::Rust, true) => vec![
                Stmt::Raw("let mut tail = [0u8; 64];".into()),
                Stmt::Raw("tail[..rem].copy_from_slice(&data[offset..]);".into()),
                Stmt::let_("block", false, slice(), Expr::raw("&tail")),
            ],
            (Target::C, false) => {
                vec![Stmt::let_(
                    "block",
                    false,
                    ptr(),
                    Expr::raw("data + offset"),
                )]
            }
            (Target::C, true) => vec![
                Stmt::Raw("uint8_t tail[64] = {0};".into()),
                Stmt::Raw(
                    "for (size_t i = 0; i < rem; i++) { tail[i] = data[offset + i]; }".into(),
                ),
                Stmt::let_("block", false, ptr(), Expr::raw("tail")),
            ],
        }
    }

    /// Drain the output mask's set bits to the sink. Rust uses the unrolled
    /// `push_indexes` scatter (reserve once + raw writes); C keeps a per-bit
    /// loop (its sink is already a raw `out[count++]` store).
    fn drain(self, mask_init: Expr) -> Vec<Stmt> {
        match self {
            Target::Rust => vec![Stmt::Expr(Expr::call(
                Expr::path("push_indexes"),
                vec![
                    mask_init,
                    Expr::cast(Expr::path("offset"), Type::name("u32")),
                    Expr::path("out"),
                ],
            ))],
            Target::C => vec![
                Stmt::let_("mask", true, Type::name("u64"), mask_init),
                Stmt::While {
                    cond: Expr::binary(BinOp::Ne, Expr::path("mask"), Expr::int(0)),
                    body: Block(vec![
                        Stmt::Raw(
                            "out[(*out_count)++] = (uint32_t) offset + __builtin_ctzll(mask);"
                                .into(),
                        ),
                        Stmt::assign_op(
                            Expr::path("mask"),
                            BinOp::BitAnd,
                            Expr::binary(BinOp::Sub, Expr::path("mask"), Expr::int(1)),
                        ),
                    ]),
                },
            ],
        }
    }

    /// Tail mask operand: `n_out & <this>` clears the zero-padding past `rem`.
    fn tail_mask_operand(self) -> Expr {
        match self {
            Target::Rust => Expr::raw("((1u64 << rem) - 1)"),
            Target::C => Expr::raw("(((uint64_t)1 << rem) - 1)"),
        }
    }

    /// The input length: Rust `data.len()`, C the `len` parameter.
    fn len(self) -> Expr {
        match self {
            Target::Rust => Expr::call(Expr::path("data.len"), vec![]),
            Target::C => Expr::path("len"),
        }
    }

    /// Wrap the lowered body in the target's `index_structurals` signature.
    fn index_func(self, name: &str, body: Vec<Stmt>) -> Func {
        match self {
            Target::Rust => Func::new(name)
                .public()
                .doc("Structural indexer — a blockwise specialization of the bitstream")
                .doc("graph with a SIMD/scalar `class_mask`; byte-equivalent to")
                .doc("`interp::run`. Generated by the typed-AST emitter (`emit::lower`).")
                .param("data", Type::slice(Type::name("u8")))
                .param("out", Type::RefMut(Box::new(Type::name("Vec<u32>"))))
                .body(body),
            Target::C => Func::new(name)
                .doc("Structural indexer in CUDA-C: writes positions to `out`.")
                .doc("Lowered from the same IR graph by the typed-AST emitter.")
                .param("data", Type::Ref(Box::new(Type::name("u8"))))
                .param("len", Type::name("usize"))
                .param("out", Type::RefMut(Box::new(Type::name("u32"))))
                .param("out_count", Type::RefMut(Box::new(Type::name("usize"))))
                .body(body),
        }
    }
}

/// The per-block statements: bind `block`, compute each live node into `nK`,
/// then drain the output mask's set bits into `out`.
fn block_body(
    target: Target,
    graph: &Graph,
    live: &[bool],
    out_id: usize,
    tail: bool,
) -> Vec<Stmt> {
    let mut b = target.block_bind(tail);
    for (i, op) in graph.nodes().iter().enumerate() {
        if live[i] {
            b.extend(node_stmts(target, i, op));
        }
    }

    // Output mask; the tail masks off the zero-padding beyond `rem`.
    let out_node = Expr::path(format!("n{out_id}"));
    let mask_init = if tail {
        Expr::binary(BinOp::BitAnd, out_node, target.tail_mask_operand())
    } else {
        out_node
    };
    b.extend(target.drain(mask_init));
    b
}

/// `let nK = …;` (plus carry updates) for node `i`, mirroring `interp::step`.
fn node_stmts(target: Target, i: usize, op: &Op) -> Vec<Stmt> {
    let n = format!("n{i}");
    let nref = |id: crate::ir::NodeId| Expr::path(format!("n{}", id.0));
    let node = |name: String, init: Expr| Stmt::Let {
        name,
        mutable: false,
        ty: target.node_ty(),
        init,
    };
    match *op {
        Op::Class(class) => {
            let members: Vec<u8> = class.members().collect();
            vec![node(n, target.class_expr(&members))]
        }
        Op::Const(pattern) => vec![Stmt::let_(n, false, Type::name("u64"), Expr::hex(pattern))],
        Op::Not(a) => vec![node(n, Expr::unary(UnOp::Not, nref(a)))],
        Op::And(a, b) => vec![node(n, Expr::binary(BinOp::BitAnd, nref(a), nref(b)))],
        Op::Or(a, b) => vec![node(n, Expr::binary(BinOp::BitOr, nref(a), nref(b)))],
        Op::Xor(a, b) => vec![node(n, Expr::binary(BinOp::BitXor, nref(a), nref(b)))],
        Op::ShiftLeft1(a) | Op::ShiftLeft1Seeded(a) => {
            let carry = format!("carry_{i}");
            vec![
                node(
                    n,
                    Expr::binary(
                        BinOp::BitOr,
                        Expr::binary(BinOp::Shl, nref(a), Expr::int(1)),
                        Expr::path(carry.clone()),
                    ),
                ),
                Stmt::assign(
                    Expr::path(carry),
                    Expr::binary(BinOp::Shr, nref(a), Expr::int(63)),
                ),
            ]
        }
        Op::PrefixXor(a) => {
            let carry = format!("carry_{i}");
            vec![
                node(
                    n.clone(),
                    Expr::binary(
                        BinOp::BitXor,
                        Expr::call(Expr::path("prefix_xor"), vec![nref(a)]),
                        Expr::path(carry.clone()),
                    ),
                ),
                // Broadcast bit 63 of the running parity to the next block.
                Stmt::assign(
                    Expr::path(carry),
                    Expr::cast(
                        Expr::binary(
                            BinOp::Shr,
                            Expr::cast(Expr::path(n), Type::name("i64")),
                            Expr::int(63),
                        ),
                        Type::name("u64"),
                    ),
                ),
            ]
        }
        Op::Add(a, b) => {
            let carry = format!("carry_{i}");
            let sum = format!("sum_{i}");
            let c1 = format!("c1_{i}");
            let c2 = format!("c2_{i}");
            let add = target.add_op();
            vec![
                node(sum.clone(), Expr::binary(add, nref(a), nref(b))),
                node(
                    c1.clone(),
                    Expr::cast(
                        Expr::binary(BinOp::Lt, Expr::path(sum.clone()), nref(a)),
                        Type::name("u64"),
                    ),
                ),
                node(
                    n.clone(),
                    Expr::binary(add, Expr::path(sum.clone()), Expr::path(carry.clone())),
                ),
                node(
                    c2.clone(),
                    Expr::cast(
                        Expr::binary(BinOp::Lt, Expr::path(n), Expr::path(sum)),
                        Type::name("u64"),
                    ),
                ),
                Stmt::assign(
                    Expr::path(carry),
                    Expr::binary(BinOp::BitOr, Expr::path(c1), Expr::path(c2)),
                ),
            ]
        }
        Op::Regions(q, s, t) => {
            let carry = format!("carry_{i}");
            vec![node(
                n,
                Expr::call(
                    Expr::path("resolve_regions"),
                    vec![nref(q), nref(s), nref(t), target.carry_ref(&carry)],
                ),
            )]
        }
    }
}

fn is_stateful(op: &Op) -> bool {
    matches!(
        op,
        Op::ShiftLeft1(_)
            | Op::ShiftLeft1Seeded(_)
            | Op::PrefixXor(_)
            | Op::Add(..)
            | Op::Regions(..)
    )
}

/// Mark every node reachable from the output — the rest are dead (e.g. the
/// bracket-nesting streams a JSON graph appends after the structural output)
/// and are not emitted.
fn live_nodes(graph: &Graph) -> Vec<bool> {
    let mut live = vec![false; graph.nodes().len()];
    let mut stack = vec![graph.output().0 as usize];
    while let Some(i) = stack.pop() {
        if live[i] {
            continue;
        }
        live[i] = true;
        for operand in operands(&graph.nodes()[i]) {
            stack.push(operand);
        }
    }
    live
}

fn operands(op: &Op) -> Vec<usize> {
    let id = |n: crate::ir::NodeId| n.0 as usize;
    match *op {
        Op::Class(_) | Op::Const(_) => vec![],
        Op::Not(a) | Op::ShiftLeft1(a) | Op::ShiftLeft1Seeded(a) | Op::PrefixXor(a) => vec![id(a)],
        Op::And(a, b) | Op::Or(a, b) | Op::Xor(a, b) | Op::Add(a, b) => vec![id(a), id(b)],
        Op::Regions(a, b, c) => vec![id(a), id(b), id(c)],
    }
}

/// The `class_mask` family — the one ISA-specific seam. AVX2 classify
/// (`cmpeq` + `movemask`) where available, with a scalar / non-x86 fallback.
/// The whole `u64` bit-algebra around it is identical to a SIMD kernel; this is
/// the only place that touches intrinsics, so it is emitted via the `Raw`
/// escape hatch (a future milestone could model intrinsics in typed AST too).
fn class_mask_helper() -> Item {
    Item::Raw(
        r#"/// 64-bit class mask for one 64-byte block: bit i set iff block[i] is in
/// `members`. AVX2 classify where available; scalar fallback otherwise.
#[cfg(target_arch = "x86_64")]
fn class_mask(block: &[u8], members: &[u8]) -> u64 {
    if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw") {
        unsafe { class_mask_avx512(block, members) }
    } else if std::is_x86_feature_detected!("avx2") {
        unsafe { class_mask_avx2(block, members) }
    } else {
        class_mask_scalar(block, members)
    }
}

/// NEON: `vceqq_u8` per member over four 16-byte lanes; a bit-position mask +
/// horizontal add packs each lane to 16 bits (NEON has no movemask).
#[cfg(target_arch = "aarch64")]
fn class_mask(block: &[u8], members: &[u8]) -> u64 {
    use core::arch::aarch64::*;
    let bits: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
    let mut mask: u64 = 0;
    unsafe {
        let bitmask = vld1q_u8(bits.as_ptr());
        let mut chunk = 0usize;
        while chunk < 4 {
            let v = vld1q_u8(block.as_ptr().add(chunk * 16));
            let mut acc = vdupq_n_u8(0);
            for &b in members {
                acc = vorrq_u8(acc, vceqq_u8(v, vdupq_n_u8(b)));
            }
            let masked = vandq_u8(acc, bitmask);
            let lo = vaddv_u8(vget_low_u8(masked)) as u64;
            let hi = vaddv_u8(vget_high_u8(masked)) as u64;
            mask |= (lo | (hi << 8)) << (chunk * 16);
            chunk += 1;
        }
    }
    mask
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn class_mask(block: &[u8], members: &[u8]) -> u64 {
    class_mask_scalar(block, members)
}

fn class_mask_scalar(block: &[u8], members: &[u8]) -> u64 {
    let mut m = 0u64;
    for i in 0..64 {
        if members.contains(&block[i]) {
            m |= 1u64 << i;
        }
    }
    m
}

/// One `cmpeq` per class member, OR'd, over the two 32-byte halves; the two
/// movemasks pack into the 64-bit block mask.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn class_mask_avx2(block: &[u8], members: &[u8]) -> u64 {
    use core::arch::x86_64::*;
    let lo = _mm256_loadu_si256(block.as_ptr() as *const __m256i);
    let hi = _mm256_loadu_si256(block.as_ptr().add(32) as *const __m256i);
    let mut acc_lo = _mm256_setzero_si256();
    let mut acc_hi = _mm256_setzero_si256();
    for &b in members {
        let v = _mm256_set1_epi8(b as i8);
        acc_lo = _mm256_or_si256(acc_lo, _mm256_cmpeq_epi8(lo, v));
        acc_hi = _mm256_or_si256(acc_hi, _mm256_cmpeq_epi8(hi, v));
    }
    let m_lo = _mm256_movemask_epi8(acc_lo) as u32 as u64;
    let m_hi = _mm256_movemask_epi8(acc_hi) as u32 as u64;
    m_lo | (m_hi << 32)
}

/// AVX-512BW: `vpcmpeqb` straight to a 64-bit mask register, OR'd per member.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn class_mask_avx512(block: &[u8], members: &[u8]) -> u64 {
    use core::arch::x86_64::*;
    let v = _mm512_loadu_si512(block.as_ptr() as *const _);
    let mut m: u64 = 0;
    for &b in members {
        m |= _mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(b as i8));
    }
    m
}"#
        .into(),
    )
}

/// The `prefix_xor` family — running parity (bit i = XOR of bits 0..=i), the
/// in-quote primitive. PCLMULQDQ (carry-less multiply by all-ones — the
/// simdjson trick the production kernels use) where available; the scalar
/// log-step shift cascade otherwise.
fn prefix_xor_helper() -> Item {
    Item::Raw(
        r#"/// Running parity (bit i = XOR of bits 0..=i). PCLMULQDQ where available;
/// scalar log-step cascade otherwise.
#[cfg(target_arch = "x86_64")]
fn prefix_xor(x: u64) -> u64 {
    if std::is_x86_feature_detected!("pclmulqdq") {
        unsafe { prefix_xor_pclmul(x) }
    } else {
        prefix_xor_scalar(x)
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn prefix_xor(x: u64) -> u64 {
    prefix_xor_scalar(x)
}

fn prefix_xor_scalar(mut x: u64) -> u64 {
    x ^= x << 1;
    x ^= x << 2;
    x ^= x << 4;
    x ^= x << 8;
    x ^= x << 16;
    x ^= x << 32;
    x
}

/// Carry-less multiply of `x` by an all-ones mask yields the prefix-XOR in the
/// low 64 bits (simdjson's quote-parity trick).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "pclmulqdq,sse2")]
unsafe fn prefix_xor_pclmul(x: u64) -> u64 {
    use core::arch::x86_64::*;
    let a = _mm_set_epi64x(0, x as i64);
    let ones = _mm_set1_epi8(-1);
    let r = _mm_clmulepi64_si128(a, ones, 0);
    _mm_cvtsi128_si64(r) as u64
}"#
        .into(),
    )
}

/// The sequential three-state region resolver for [`Op::Regions`], emitted
/// verbatim via the `Raw` escape hatch (mirrors `interp::resolve_regions`).
/// A future milestone could model this in typed AST too.
fn region_helpers() -> Item {
    Item::Raw(
        r#"/// Three-state region resolution (normal/quote/comment); `state` carries
/// the region state across blocks. Mirrors `interp::resolve_regions`.
fn resolve_regions(q: u64, s: u64, n: u64, state: &mut u64) -> u64 {
    const NORMAL: u64 = 0;
    const QUOTE: u64 = 1;
    const COMMENT: u64 = 2;
    let mut inert = 0u64;
    let mut run_start = 0u32;
    let mut events = q | s | n;
    while events != 0 {
        let p = events.trailing_zeros();
        let bit = 1u64 << p;
        match *state {
            QUOTE => {
                if q & bit != 0 {
                    inert |= range_mask(run_start, p);
                    *state = NORMAL;
                }
            }
            COMMENT => {
                if n & bit != 0 {
                    inert |= range_mask(run_start, p);
                    *state = NORMAL;
                }
            }
            _ => {
                if q & bit != 0 {
                    *state = QUOTE;
                    run_start = p;
                } else if s & bit != 0 {
                    *state = COMMENT;
                    run_start = p;
                }
            }
        }
        events &= events - 1;
    }
    if *state != NORMAL {
        inert |= range_mask(run_start, 64);
    }
    inert
}

fn range_mask(from: u32, to: u32) -> u64 {
    let hi = if to >= 64 { !0u64 } else { (1u64 << to) - 1 };
    hi & !((1u64 << from) - 1)
}"#
        .into(),
    )
}

// ---- Whole-loop-inline x86 variants: production-parity structural index. ----
//
// `index_structurals` dispatches to a baked `#[target_feature]` whole-loop
// variant on x86 (AVX-512 → AVX2): one block load reused across all classes,
// inline PCLMULQDQ quote-parity, the shared `push_indexes` scatter, two blocks
// per iteration — so the whole per-block pipeline inlines (the production
// kernel's structure). Non-x86 / pre-AVX2 CPUs use the portable `class_mask`
// path (same scatter), which keeps the real NEON classify on aarch64. Graphs
// that use the sequential `Regions` resolver stay on the portable path.

#[derive(Clone, Copy)]
enum IsaInline {
    Avx512,
    Avx2,
    Neon,
}

/// The full Rust index surface for one graph: a `pub fn <name>` dispatcher plus
/// its baked x86 variants and the portable fallback. Graphs using
/// [`Op::Regions`] (not covered by the baked path) get the portable indexer
/// alone, under `name`.
pub fn rust_index_items(graph: &Graph, name: &str) -> Vec<Item> {
    if graph.nodes().iter().any(|o| matches!(o, Op::Regions(..))) {
        return vec![emit_index(Target::Rust, graph, name)];
    }
    let live = live_nodes(graph);
    vec![
        index_dispatch(graph, name),
        baked_variant(IsaInline::Avx512, graph, &live, name),
        baked_variant(IsaInline::Avx2, graph, &live, name),
        baked_variant(IsaInline::Neon, graph, &live, name),
        emit_index(Target::Rust, graph, &format!("{name}_portable")),
    ]
}

/// The `pub fn <name>` dispatcher: pick the best baked variant at runtime
/// (x86: AVX-512 -> AVX2; aarch64: NEON), else the portable fallback.
fn index_dispatch(graph: &Graph, name: &str) -> Item {
    let has_px = graph.nodes().iter().any(|o| matches!(o, Op::PrefixXor(_)));
    // x86 quote-parity needs PCLMULQDQ; aarch64's PMULL needs the `aes` feature.
    let pclmul = if has_px {
        " && std::is_x86_feature_detected!(\"pclmulqdq\")"
    } else {
        ""
    };
    let aes = if has_px {
        " && std::arch::is_aarch64_feature_detected!(\"aes\")"
    } else {
        ""
    };
    Item::Raw(format!(
        r#"/// Structural indexer: dispatches to a baked AVX-512/AVX2/NEON whole-loop
/// variant, else the portable `class_mask` path. Generated by `emit::lower`.
pub fn {name}(data: &[u8], out: &mut Vec<u32>) {{
    #[cfg(target_arch = "x86_64")]
    {{
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw"){pclmul} {{
            // SAFETY: the required features were just detected.
            return unsafe {{ {name}_avx512(data, out) }};
        }}
        if std::is_x86_feature_detected!("avx2"){pclmul} {{
            // SAFETY: the required features were just detected.
            return unsafe {{ {name}_avx2(data, out) }};
        }}
    }}
    #[cfg(target_arch = "aarch64")]
    {{
        if std::arch::is_aarch64_feature_detected!("neon"){aes} {{
            // SAFETY: the required features were just detected.
            return unsafe {{ {name}_neon(data, out) }};
        }}
    }}
    {name}_portable(data, out)
}}"#
    ))
}

/// One baked `#[target_feature]` whole-loop variant for `isa`: the entire
/// per-block pipeline inlined, two blocks per iteration.
fn baked_variant(isa: IsaInline, graph: &Graph, live: &[bool], name: &str) -> Item {
    use std::fmt::Write;
    let out_id = graph.output().0;
    let has_px = graph.nodes().iter().any(|o| matches!(o, Op::PrefixXor(_)));
    let (cfg_arch, use_path, suffix, feats) = match isa {
        IsaInline::Avx512 if has_px => ("x86_64", "x86_64", "avx512", "avx512f,avx512bw,pclmulqdq"),
        IsaInline::Avx512 => ("x86_64", "x86_64", "avx512", "avx512f,avx512bw"),
        IsaInline::Avx2 if has_px => ("x86_64", "x86_64", "avx2", "avx2,pclmulqdq"),
        IsaInline::Avx2 => ("x86_64", "x86_64", "avx2", "avx2"),
        IsaInline::Neon if has_px => ("aarch64", "aarch64", "neon", "neon,aes"),
        IsaInline::Neon => ("aarch64", "aarch64", "neon", "neon"),
    };
    // Only the x86 PCLMULQDQ path hoists an all-ones constant; NEON's PMULL
    // prefix-xor is a helper call.
    let preamble = if has_px && !matches!(isa, IsaInline::Neon) {
        "    let ones = _mm_set1_epi8(-1);\n"
    } else {
        ""
    };
    let mut carries = String::new();
    let mut nodes = String::new();
    for (i, op) in graph.nodes().iter().enumerate() {
        if !live[i] {
            continue;
        }
        if is_stateful(op) {
            let init = u64::from(matches!(op, Op::ShiftLeft1Seeded(_)));
            let _ = writeln!(carries, "    let mut carry_{i}: u64 = {init};");
        }
        nodes.push_str(&baked_node_line(isa, i, op));
    }
    let load0 = baked_load(isa, "data.as_ptr().add(offset)");
    let load1 = baked_load(isa, "data.as_ptr().add(offset + 64)");
    let loadt = baked_load(isa, "tail.as_ptr()");
    Item::Raw(format!(
        r#"#[cfg(target_arch = "{cfg_arch}")]
#[target_feature(enable = "{feats}")]
unsafe fn {name}_{suffix}(data: &[u8], out: &mut Vec<u32>) {{
    use core::arch::{use_path}::*;
{preamble}{carries}    let len = data.len();
    let mut offset = 0usize;
    // Two blocks per iteration: amortizes loop control and overlaps block 1's
    // classify with block 0's extract.
    while offset + 128 <= len {{
{load0}{nodes}        push_indexes(n{out_id}, offset as u32, out);
{load1}{nodes}        push_indexes(n{out_id}, (offset + 64) as u32, out);
        offset += 128;
    }}
    while offset + 64 <= len {{
{load0}{nodes}        push_indexes(n{out_id}, offset as u32, out);
        offset += 64;
    }}
    let rem = len - offset;
    if rem > 0 {{
        let mut tail = [0u8; 64];
        tail[..rem].copy_from_slice(&data[offset..]);
{loadt}{nodes}        push_indexes(n{out_id} & ((1u64 << rem) - 1), offset as u32, out);
    }}
}}"#
    ))
}

/// The block load for `isa`: AVX-512 one register `v`; AVX2 two halves `lo`/`hi`;
/// NEON four 16-byte lanes `b0..b3`.
fn baked_load(isa: IsaInline, ptr: &str) -> String {
    match isa {
        IsaInline::Avx512 => format!("        let v = _mm512_loadu_si512({ptr} as *const _);\n"),
        IsaInline::Avx2 => format!(
            "        let lo = _mm256_loadu_si256({ptr} as *const __m256i);\n        let hi = _mm256_loadu_si256(({ptr}).add(32) as *const __m256i);\n"
        ),
        IsaInline::Neon => format!(
            "        let b0 = vld1q_u8({ptr});\n        let b1 = vld1q_u8(({ptr}).add(16));\n        let b2 = vld1q_u8(({ptr}).add(32));\n        let b3 = vld1q_u8(({ptr}).add(48));\n"
        ),
    }
}

/// One node's inline statement(s) for a baked x86 variant.
fn baked_node_line(isa: IsaInline, i: usize, op: &Op) -> String {
    let nr = |id: crate::ir::NodeId| format!("n{}", id.0);
    match *op {
        Op::Class(class) => {
            let members: Vec<u8> = class.members().collect();
            format!("        let n{i} = {};\n", baked_classify(isa, &members))
        }
        Op::Const(p) => format!("        let n{i}: u64 = 0x{p:x};\n"),
        Op::Not(a) => format!("        let n{i} = !{};\n", nr(a)),
        Op::And(a, b) => format!("        let n{i} = {} & {};\n", nr(a), nr(b)),
        Op::Or(a, b) => format!("        let n{i} = {} | {};\n", nr(a), nr(b)),
        Op::Xor(a, b) => format!("        let n{i} = {} ^ {};\n", nr(a), nr(b)),
        Op::ShiftLeft1(a) | Op::ShiftLeft1Seeded(a) => format!(
            "        let n{i} = ({a} << 1) | carry_{i};\n        carry_{i} = {a} >> 63;\n",
            a = nr(a)
        ),
        Op::PrefixXor(a) => {
            let a = nr(a);
            let px = match isa {
                IsaInline::Neon => format!("neon_ops::prefix_xor({a})"),
                _ => format!(
                    "(_mm_cvtsi128_si64(_mm_clmulepi64_si128(_mm_set_epi64x(0, {a} as i64), ones, 0)) as u64)"
                ),
            };
            format!(
                "        let n{i} = {px} ^ carry_{i};\n        carry_{i} = ((n{i} as i64) >> 63) as u64;\n"
            )
        }
        Op::Add(a, b) => format!(
            "        let sum_{i} = {a}.wrapping_add({b});\n        let c1_{i} = (sum_{i} < {a}) as u64;\n        let n{i} = sum_{i}.wrapping_add(carry_{i});\n        let c2_{i} = (n{i} < sum_{i}) as u64;\n        carry_{i} = c1_{i} | c2_{i};\n",
            a = nr(a),
            b = nr(b)
        ),
        Op::Regions(..) => "        // Regions: handled by the portable fallback\n".to_string(),
    }
}

/// Inline class mask for `members`, reusing the loaded register(s).
fn baked_classify(isa: IsaInline, members: &[u8]) -> String {
    match isa {
        IsaInline::Avx512 => members
            .iter()
            .map(|b| format!("_mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8({}))", *b as i8))
            .collect::<Vec<_>>()
            .join(" | "),
        IsaInline::Avx2 => {
            let fold = |reg: &str| {
                members
                    .iter()
                    .map(|b| format!("_mm256_cmpeq_epi8({reg}, _mm256_set1_epi8({}))", *b as i8))
                    .reduce(|a, c| format!("_mm256_or_si256({a}, {c})"))
                    .unwrap()
            };
            format!(
                "(_mm256_movemask_epi8({lo}) as u32 as u64) | ((_mm256_movemask_epi8({hi}) as u32 as u64) << 32)",
                lo = fold("lo"),
                hi = fold("hi")
            )
        }
        IsaInline::Neon => members
            .iter()
            .map(|b| format!("neon_ops::eq_mask(b0, b1, b2, b3, {b}u8)"))
            .collect::<Vec<_>>()
            .join(" | "),
    }
}
