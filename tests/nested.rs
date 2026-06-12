//! Differential tests for the nested tape (`parse_nested`) of the generated
//! JSON kernel: structure must agree with serde_json on randomized documents,
//! bracket matching must be exact across 64-byte block seams, and malformed
//! input must report first-error positions without panicking.

use falx::kernels::json;
use serde_json::Value;

/// xorshift64* RNG; avoids a dev-dependency for test data generation.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

/// Strings that stress the quote/escape machinery: structural bytes inside
/// strings, backslash runs (odd and even), escaped quotes, trailing escapes.
const STRING_POOL: &[&str] = &[
    "plain",
    "",
    "with space",
    "br{ack}ets",
    "[inside]",
    "a,b:c,d",
    "quote\"inside",
    "back\\slash",
    "run\\\\\\of\\\\backslashes",
    "ends with \\",
    "\\",
    "\\\\",
    "tab\tand newline\n",
    "unicode \u{00e9}\u{6587}\u{1F980}",
    "{\"looks\":[\"like\",\"json\"]}",
];

fn gen_string(rng: &mut Rng) -> String {
    STRING_POOL[(rng.next() % STRING_POOL.len() as u64) as usize].to_string()
}

fn gen_value(rng: &mut Rng, depth: usize) -> Value {
    let choice = if depth == 0 {
        rng.next() % 5
    } else {
        rng.next() % 7
    };
    match choice {
        0 => Value::Null,
        1 => Value::Bool(rng.next().is_multiple_of(2)),
        2 => Value::from(rng.next() as i64 % 100_000),
        3 => Value::from((rng.next() as i64 % 100_000) as f64 / 128.0),
        4 => Value::String(gen_string(rng)),
        5 => {
            let n = (rng.next() % 5) as usize;
            Value::Array((0..n).map(|_| gen_value(rng, depth - 1)).collect())
        }
        _ => {
            let n = (rng.next() % 5) as usize;
            // Unique keys: duplicate keys would collapse in serde's map and
            // desynchronize the pairwise walk below.
            let map = (0..n)
                .map(|i| {
                    (
                        format!("k{i} {}", gen_string(rng)),
                        gen_value(rng, depth - 1),
                    )
                })
                .collect();
            Value::Object(map)
        }
    }
}

/// Assert that a falx node and a serde value have identical structure:
/// containers match kind and arity, scalar spans re-parse to equal values.
fn assert_matches(node: &json::Node, value: &Value, context: &str) {
    match value {
        Value::Object(map) => {
            assert_eq!(node.open(), Some(b'{'), "{context}: expected object");
            let items: Vec<json::Node> = node.items().collect();
            assert_eq!(
                items.len(),
                2 * map.len(),
                "{context}: object item count (keys and values alternate)"
            );
            // serde's default map is sorted; serialization order equals
            // iteration order, and falx yields input order, so zip works.
            for (i, (key, val)) in map.iter().enumerate() {
                let key_node = &items[2 * i];
                assert_eq!(key_node.open(), None, "{context}: key must be scalar");
                let parsed: String = serde_json::from_slice(key_node.bytes())
                    .unwrap_or_else(|e| panic!("{context}: key span unparseable: {e}"));
                assert_eq!(&parsed, key, "{context}: key mismatch");
                assert_matches(&items[2 * i + 1], val, context);
            }
        }
        Value::Array(arr) => {
            assert_eq!(node.open(), Some(b'['), "{context}: expected array");
            let items: Vec<json::Node> = node.items().collect();
            assert_eq!(items.len(), arr.len(), "{context}: array length");
            for (item, val) in items.iter().zip(arr) {
                assert_matches(item, val, context);
            }
        }
        scalar => {
            assert_eq!(
                node.open(),
                None,
                "{context}: expected scalar, got container {:?}",
                String::from_utf8_lossy(node.bytes())
            );
            let parsed: Value = serde_json::from_slice(node.bytes())
                .unwrap_or_else(|e| panic!("{context}: scalar span unparseable: {e}"));
            assert_eq!(&parsed, scalar, "{context}: scalar value");
        }
    }
}

fn assert_doc_matches(text: &str, value: &Value, context: &str) {
    let doc = json::parse_nested(text.as_bytes());
    assert_eq!(doc.error, None, "{context}: unexpected nest error");
    let items: Vec<json::Node> = doc.items().collect();
    assert_eq!(items.len(), 1, "{context}: expected one top-level value");
    assert_matches(&items[0], value, context);
}

#[test]
fn differential_vs_serde_compact() {
    let mut rng = Rng(0xBEEF_CAFE_1234_5678);
    for round in 0..400 {
        let value = gen_value(&mut rng, 5);
        let text = serde_json::to_string(&value).unwrap();
        assert_doc_matches(&text, &value, &format!("compact round {round}"));
    }
}

#[test]
fn differential_vs_serde_pretty() {
    let mut rng = Rng(0x00DD_BA11_F00D_5EED);
    for round in 0..200 {
        let value = gen_value(&mut rng, 5);
        let text = serde_json::to_string_pretty(&value).unwrap();
        assert_doc_matches(&text, &value, &format!("pretty round {round}"));
    }
}

/// Slide a fixed document across every 64-byte block alignment so quotes,
/// escapes, and brackets land on (and straddle) every seam position.
#[test]
fn block_seam_sweep() {
    let value: Value = serde_json::from_str(
        r#"{"k\\\\": ["va\"l", -3.5, {"x": [], "y\\": [null, true]}], "": {}}"#,
    )
    .unwrap();
    let text = serde_json::to_string(&value).unwrap();
    for pad in 0..130 {
        let padded = format!("{}{}", " ".repeat(pad), text);
        assert_doc_matches(&padded, &value, &format!("pad {pad}"));
    }
}

#[test]
fn multiple_top_level_documents() {
    let data = b" {\"a\": 1} [2, 3]\n{\"b\": [4]} 5 ";
    let doc = json::parse_nested(data);
    assert_eq!(doc.error, None);
    let items: Vec<json::Node> = doc.items().collect();
    assert_eq!(items.len(), 4);
    assert_eq!(items[0].open(), Some(b'{'));
    assert_eq!(items[1].open(), Some(b'['));
    assert_eq!(items[2].open(), Some(b'{'));
    assert_eq!(items[3].open(), None);
    assert_eq!(items[3].bytes(), b"5");
}

#[test]
fn empty_and_scalar_inputs() {
    for empty in [&b""[..], b"   ", b" \t\r\n "] {
        let doc = json::parse_nested(empty);
        assert_eq!(doc.error, None);
        assert_eq!(doc.items().count(), 0);
        assert_eq!(doc.tape().len(), 0);
    }
    let doc = json::parse_nested(b" 42 ");
    assert_eq!(doc.error, None);
    let items: Vec<json::Node> = doc.items().collect();
    assert_eq!(items.len(), 1);
    assert_eq!(items[0].bytes(), b"42");
}

#[test]
fn empty_containers() {
    let value: Value = serde_json::from_str(r#"[[], {}, [{}, [[]]]]"#).unwrap();
    assert_doc_matches(r#"[[], {}, [{}, [[]]]]"#, &value, "empty containers");
}

/// Bracket matching is a heap stack: deep nesting must neither overflow nor
/// mismatch. Navigation here is iterative on purpose (no recursive dump).
#[test]
fn deep_nesting() {
    const DEPTH: usize = 100_000;
    let mut text = "[".repeat(DEPTH);
    text.push_str(&"]".repeat(DEPTH));
    let doc = json::parse_nested(text.as_bytes());
    assert_eq!(doc.error, None);
    // The outermost pair: first tape entry matches the last.
    let tape = doc.tape();
    assert_eq!(tape.len(), 2 * DEPTH);
    assert_eq!((tape[0] >> 32) as usize, tape.len() - 1);
    assert_eq!((tape[tape.len() - 1] >> 32) as usize, 0);
    // Walk down without recursion.
    let mut node = doc.items().next().unwrap();
    let mut depth = 1;
    while let Some(child) = node.items().next() {
        assert_eq!(child.open(), Some(b'['));
        node = child;
        depth += 1;
    }
    assert_eq!(depth, DEPTH);
}

#[test]
fn error_positions() {
    let cases: &[(&[u8], json::NestError)] = &[
        (b"{\"a\": [1, 2}", json::NestError::UnmatchedClose(11)),
        (b"[1, 2", json::NestError::UnclosedOpen(0)),
        (b"1]", json::NestError::UnmatchedClose(1)),
        (b"}", json::NestError::UnmatchedClose(0)),
        (b"[[]", json::NestError::UnclosedOpen(0)),
        (b"[ {", json::NestError::UnclosedOpen(2)),
    ];
    for (data, expected) in cases {
        let doc = json::parse_nested(data);
        assert_eq!(
            doc.error,
            Some(*expected),
            "input {:?}",
            String::from_utf8_lossy(data)
        );
    }
}

#[test]
fn brackets_inside_strings_are_inert() {
    let doc = json::parse_nested(br#"["a}b]{", "\"["]"#);
    assert_eq!(doc.error, None);
    let arr = doc.items().next().unwrap();
    let kids: Vec<&[u8]> = arr.items().map(|n| n.bytes()).collect();
    assert_eq!(kids, [&br#""a}b]{""#[..], &br#""\"[""#[..]]);
}

/// Every bracket entry's partner reference must be mutual, and partners
/// must enclose strictly nested position ranges.
#[test]
fn tape_partner_invariants() {
    let mut rng = Rng(0x7777_AAAA_3333_DDDD);
    for _ in 0..100 {
        let value = gen_value(&mut rng, 6);
        let text = serde_json::to_string(&value).unwrap();
        let doc = json::parse_nested(text.as_bytes());
        let tape = doc.tape();
        for (i, &entry) in tape.iter().enumerate() {
            let pos = (entry as u32) as usize;
            let partner = (entry >> 32) as u32;
            match text.as_bytes()[pos] {
                b'{' | b'[' => {
                    assert_ne!(partner, u32::MAX, "open at {pos} unmatched");
                    let back = tape[partner as usize];
                    assert_eq!((back >> 32) as usize, i, "partner not mutual");
                    assert!(partner as usize > i, "close precedes open");
                }
                b'}' | b']' => {
                    let back = tape[partner as usize];
                    assert_eq!((back >> 32) as u32 as usize, i, "partner not mutual");
                }
                _ => {}
            }
        }
    }
}

/// The spec front-end must produce the identical kernel: specs/json.toml
/// (which auto-appends nesting bytes to the structural set) and the in-tree
/// json_dialect() differ in structural byte order, which a CharClass bitmap
/// must erase.
#[cfg(feature = "spec")]
#[test]
fn json_spec_matches_checked_in_kernel() {
    let toml_text =
        std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/specs/json.toml")).unwrap();
    let spec = falx::spec::parse(&toml_text).unwrap();
    let emitted =
        falx::codegen::emit_parser_with_columns(&spec.dialect, &spec.name, &spec.columns).unwrap();
    let checked_in =
        std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/kernels/json.rs"))
            .unwrap();
    assert_eq!(emitted, checked_in);
}

/// The parallel path must produce a byte-identical tape and equal error to
/// serial for every thread count — entry carries (escapes and quote state
/// crossing chunk boundaries) come from the carry-replay prepass.
#[test]
fn parallel_matches_serial() {
    let mut rng = Rng(0x1357_9BDF_2468_ACE0);
    for round in 0..15 {
        let value = gen_value(&mut rng, 6);
        let text = vec![serde_json::to_string(&value).unwrap(); 300].join(" \n");
        let serial = json::parse_nested(text.as_bytes());
        for threads in [1, 2, 3, 5, 8, 16] {
            let par = json::parse_nested_par(text.as_bytes(), threads);
            assert_eq!(par.error, serial.error, "round {round} threads {threads}");
            assert_eq!(par.tape(), serial.tape(), "round {round} threads {threads}");
        }
    }
}

/// Containers spanning every chunk force the residue merge to do real
/// work: almost every bracket is unmatched locally.
#[test]
fn parallel_cross_chunk_nesting() {
    const N: usize = 50_000;
    let mut text = String::with_capacity(N * 16);
    text.push('[');
    for i in 0..N {
        if i > 0 {
            text.push(',');
        }
        text.push_str("[{\"k\": [1, 2]}]");
    }
    text.push(']');
    let serial = json::parse_nested(text.as_bytes());
    assert_eq!(serial.error, None);
    for threads in [2, 4, 16] {
        let par = json::parse_nested_par(text.as_bytes(), threads);
        assert_eq!(par.error, None, "threads {threads}");
        assert_eq!(par.tape(), serial.tape(), "threads {threads}");
    }

    // Pure depth: every chunk is all-residue, the merge stack peaks at N.
    let mut deep = "[".repeat(200_000);
    deep.push_str(&"]".repeat(200_000));
    let serial = json::parse_nested(deep.as_bytes());
    assert_eq!(serial.error, None);
    for threads in [2, 7, 16] {
        let par = json::parse_nested_par(deep.as_bytes(), threads);
        assert_eq!(par.error, serial.error, "deep threads {threads}");
        assert_eq!(par.tape(), serial.tape(), "deep threads {threads}");
    }
}

/// Escape runs and quoted regions crossing chunk boundaries: the carry
/// replay must hand each chunk the exact kernel state.
#[test]
fn parallel_boundary_state() {
    // Long strings of backslashes and bracket-bearing text ensure quoted
    // regions and escape runs straddle every chunk boundary at some
    // thread count.
    let unit = r#"{"a": "text [with] {brackets} and \\\\ runs \" inside", "b": [1]}"#;
    let text = vec![unit; 4000].join("\n");
    let serial = json::parse_nested(text.as_bytes());
    assert_eq!(serial.error, None);
    for threads in 2..=17 {
        let par = json::parse_nested_par(text.as_bytes(), threads);
        assert_eq!(par.error, None, "threads {threads}");
        assert_eq!(par.tape(), serial.tape(), "threads {threads}");
    }
}

/// Malformed input: the parallel path falls back to serial, so results
/// must be exactly equal, including tape truncation at the error.
#[test]
fn parallel_malformed_matches_serial() {
    let filler = "{\"k\": [1, 2, 3]} ".repeat(5_000);
    let cases = [
        format!("{filler}]{filler}"),        // unmatched close mid-stream
        format!("{filler}[ {filler}"),       // unclosed open
        format!("{filler}[1, 2}} {filler}"), // kind mismatch
        "]".to_string(),                     // tiny error input
    ];
    for (i, text) in cases.iter().enumerate() {
        let serial = json::parse_nested(text.as_bytes());
        assert!(serial.error.is_some(), "case {i} should be malformed");
        for threads in [2, 8] {
            let par = json::parse_nested_par(text.as_bytes(), threads);
            assert_eq!(par.error, serial.error, "case {i} threads {threads}");
            assert_eq!(par.tape(), serial.tape(), "case {i} threads {threads}");
        }
    }
}
