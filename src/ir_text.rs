//! The stable textual form of a falx IR module — the interchange format
//! between a frontend and the code generator.
//!
//! A [`Module`] is everything the backend needs: the lowered bitstream
//! [`Graph`], the node *roles* that give three of its streams meaning
//! (structural output, record terminators, and bracket open/close for nesting
//! dialects), and the surface attributes the generated span/column APIs are
//! built from (quote and escape convention, comment byte, record grouping, and
//! the projected columns).
//!
//! The point of writing it down is that the code generator can then be reached
//! without going through `formats`/`spec` at all: any producer that can emit
//! this text gets every backend falx has. `falx emit-ir` prints it, `falx
//! build-ir` consumes it, and the round-trip is checked for every in-tree
//! format — parse(print(m)) must regenerate byte-identical code.
//!
//! ```text
//! falx-ir 1
//! format csv
//! structural 2c,0a
//! quote 22
//! escape none
//! %0 = class 0a
//! %1 = class 2c,0a
//! %2 = class 22
//! %3 = prefix-xor %2
//! %4 = not %3
//! %5 = and %1 %4
//! output %5
//! terminators %0
//! ```
//!
//! Stability: the `falx-ir <version>` header is the compatibility contract.
//! Within a version, accepted text keeps its meaning; a breaking change to any
//! directive or opcode bumps [`IR_VERSION`].

use std::fmt::Write as _;

use crate::codegen::{Column, ColumnType};
use crate::formats::{Dialect, Escape};
use crate::framing::{Counts, Endian, Framing, Width};
use crate::ir::{CharClass, Graph, NodeId, Op};

/// Version of the textual IR this build reads and writes.
pub const IR_VERSION: u32 = 1;

/// A complete, backend-ready parser description.
///
/// This is the unit [`crate::codegen::lower`] produces and
/// [`crate::codegen::emit_module`] consumes; the text format is exactly its
/// serialization.
#[derive(Clone, Debug)]
pub struct Module {
    /// Format name; becomes the generated module's identity.
    pub name: String,
    /// Surface conventions the span and column APIs are generated from.
    pub dialect: Dialect,
    /// Projected typed columns (empty = no columnar API).
    pub columns: Vec<Column>,
    /// The lowered bitstream graph. Its designated output marks structural
    /// positions.
    pub graph: Graph,
    /// Node whose set bits are record terminators (ANDed with the output).
    pub terminators: NodeId,
    /// Live open/close bracket streams for nesting dialects.
    pub nest: Option<(NodeId, NodeId)>,
    /// Outer length-prefixed container, when the format has one. The graph
    /// describes the payload grammar; this describes how payloads are found.
    pub framing: Option<Framing>,
}

/// A textual-IR read error, with the offending line number where known.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IrError(pub String);

impl std::fmt::Display for IrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for IrError {}

fn err(line: usize, msg: impl AsRef<str>) -> IrError {
    IrError(format!("line {line}: {}", msg.as_ref()))
}

// --- printing --------------------------------------------------------------

/// Render `module` as textual IR. The output is stable for a given module:
/// same module, same bytes.
pub fn print(module: &Module) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "falx-ir {IR_VERSION}");
    let _ = writeln!(out, "format {}", module.name);
    let _ = writeln!(
        out,
        "structural {}",
        print_bytes(&module.dialect.structural)
    );
    if let Some(quote) = module.dialect.quote {
        let _ = writeln!(out, "quote {:02x}", quote);
    }
    match module.dialect.escape {
        Escape::None => {
            let _ = writeln!(out, "escape none");
        }
        Escape::Backslash(byte) => {
            let _ = writeln!(out, "escape backslash {:02x}", byte);
        }
    }
    if let Some(comment) = module.dialect.comment {
        let _ = writeln!(out, "comment {:02x}", comment);
    }
    if !module.dialect.nesting.is_empty() {
        let pairs: Vec<String> = module
            .dialect
            .nesting
            .iter()
            .map(|&(open, close)| format!("{open:02x}:{close:02x}"))
            .collect();
        let _ = writeln!(out, "nesting {}", pairs.join(","));
    }
    if module.dialect.lines_per_record != 1 {
        let _ = writeln!(out, "lines-per-record {}", module.dialect.lines_per_record);
    }
    if let Some(f) = &module.framing {
        let _ = write!(
            out,
            "frame header={} length-at={} width={} endian={} counts={} adjust={} trailer={}",
            f.header,
            f.length_at,
            f.width.as_str(),
            match f.endian {
                Endian::Le => "le",
                Endian::Be => "be",
            },
            match f.counts {
                Counts::Total => "total",
                Counts::Payload => "payload",
            },
            f.adjust,
            f.trailer,
        );
        if let Some((offset, bytes)) = &f.magic {
            let _ = write!(out, " magic={offset}:{}", print_bytes(bytes));
        }
        if f.skip_empty {
            let _ = write!(out, " skip-empty=true");
        }
        out.push('\n');
    }
    for column in &module.columns {
        let ty = match column.ty {
            ColumnType::I64 => "i64",
            ColumnType::F64 => "f64",
            ColumnType::Str => "string",
            ColumnType::Bytes => "bytes",
        };
        let _ = write!(out, "column {} {} {ty}", column.index, column.field_name());
        if let Some(key) = &column.info_key {
            let _ = write!(out, " info={key}");
        }
        out.push('\n');
    }
    for (i, op) in module.graph.nodes().iter().enumerate() {
        let _ = writeln!(out, "%{i} = {}", print_op(op));
    }
    let _ = writeln!(out, "output %{}", module.graph.output().index());
    let _ = writeln!(out, "terminators %{}", module.terminators.index());
    if let Some((open, close)) = module.nest {
        let _ = writeln!(out, "nest-open %{}", open.index());
        let _ = writeln!(out, "nest-close %{}", close.index());
    }
    out
}

fn print_op(op: &Op) -> String {
    match *op {
        Op::Class(class) => format!("class {}", print_class(&class)),
        Op::Const(pattern) => format!("const {pattern:#018x}"),
        Op::Not(a) => format!("not %{}", a.index()),
        Op::And(a, b) => format!("and %{} %{}", a.index(), b.index()),
        Op::Or(a, b) => format!("or %{} %{}", a.index(), b.index()),
        Op::Xor(a, b) => format!("xor %{} %{}", a.index(), b.index()),
        Op::ShiftLeft1(a) => format!("shl1 %{}", a.index()),
        Op::ShiftLeft1Seeded(a) => format!("shl1-seeded %{}", a.index()),
        Op::Regions(q, s, n) => format!("regions %{} %{} %{}", q.index(), s.index(), n.index()),
        Op::PrefixXor(a) => format!("prefix-xor %{}", a.index()),
        Op::Add(a, b) => format!("add %{} %{}", a.index(), b.index()),
    }
}

/// Byte sets print as comma-separated hex, with runs of three or more
/// collapsed to `lo-hi` so a 256-member class stays one short token.
fn print_class(class: &CharClass) -> String {
    let members: Vec<u8> = class.members().collect();
    if members.is_empty() {
        return "-".to_string();
    }
    let mut parts: Vec<String> = Vec::new();
    let mut i = 0;
    while i < members.len() {
        let start = members[i];
        let mut end = start;
        while i + 1 < members.len() && members[i + 1] == end + 1 {
            i += 1;
            end = members[i];
        }
        if end as u16 >= start as u16 + 2 {
            parts.push(format!("{start:02x}-{end:02x}"));
        } else if end == start {
            parts.push(format!("{start:02x}"));
        } else {
            parts.push(format!("{start:02x}"));
            parts.push(format!("{end:02x}"));
        }
        i += 1;
    }
    parts.join(",")
}

fn print_bytes(bytes: &[u8]) -> String {
    if bytes.is_empty() {
        return "-".to_string();
    }
    bytes
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<Vec<_>>()
        .join(",")
}

// --- parsing ---------------------------------------------------------------

/// Read textual IR back into a [`Module`].
///
/// Operands must refer to already-defined nodes (`%k` with k < the defining
/// index), which keeps a parsed graph in topological order by construction —
/// the same invariant the in-memory builder enforces.
pub fn parse(text: &str) -> Result<Module, IrError> {
    let mut name: Option<String> = None;
    let mut structural: Option<Vec<u8>> = None;
    let mut quote: Option<u8> = None;
    let mut escape = Escape::None;
    let mut comment: Option<u8> = None;
    let mut nesting: Vec<(u8, u8)> = Vec::new();
    let mut lines_per_record: u32 = 1;
    let mut columns: Vec<Column> = Vec::new();
    let mut graph = Graph::new();
    let mut output: Option<NodeId> = None;
    let mut terminators: Option<NodeId> = None;
    let mut nest_open: Option<NodeId> = None;
    let mut nest_close: Option<NodeId> = None;
    let mut framing: Option<Framing> = None;
    let mut version_seen = false;
    let mut next_node = 0usize;

    for (lineno, raw) in text.lines().enumerate() {
        let lineno = lineno + 1;
        // `;` starts a comment; blank lines are ignored.
        let line = match raw.find(';') {
            Some(pos) => &raw[..pos],
            None => raw,
        }
        .trim();
        if line.is_empty() {
            continue;
        }
        let mut words = line.split_whitespace();
        let head = words.next().expect("non-empty line has a first word");

        if !version_seen {
            if head != "falx-ir" {
                return Err(err(
                    lineno,
                    "expected a `falx-ir <version>` header as the first directive",
                ));
            }
            let version: u32 = words
                .next()
                .ok_or_else(|| err(lineno, "`falx-ir` needs a version number"))?
                .parse()
                .map_err(|_| err(lineno, "version must be an integer"))?;
            if version != IR_VERSION {
                return Err(err(
                    lineno,
                    format!("unsupported IR version {version}; this build reads {IR_VERSION}"),
                ));
            }
            version_seen = true;
            continue;
        }

        if let Some(rest) = head.strip_prefix('%') {
            // Node definition: `%k = op operands...`
            let index: usize = rest
                .parse()
                .map_err(|_| err(lineno, format!("bad node name `%{rest}`")))?;
            if index != next_node {
                return Err(err(
                    lineno,
                    format!("nodes must be numbered consecutively; expected %{next_node}"),
                ));
            }
            if words.next() != Some("=") {
                return Err(err(lineno, "expected `=` after the node name"));
            }
            let opcode = words
                .next()
                .ok_or_else(|| err(lineno, "expected an opcode"))?;
            let operands: Vec<&str> = words.collect();
            let id = parse_op(&mut graph, opcode, &operands, index, lineno)?;
            debug_assert_eq!(id.index(), index);
            next_node += 1;
            continue;
        }

        match head {
            "format" => {
                let value = words
                    .next()
                    .ok_or_else(|| err(lineno, "`format` needs a name"))?;
                name = Some(value.to_string());
            }
            "structural" => {
                let value = words
                    .next()
                    .ok_or_else(|| err(lineno, "`structural` needs a byte list"))?;
                structural = Some(parse_byte_list(value, lineno)?);
            }
            "quote" => quote = Some(parse_byte(words.next(), lineno, "quote")?),
            "comment" => comment = Some(parse_byte(words.next(), lineno, "comment")?),
            "escape" => {
                match words
                    .next()
                    .ok_or_else(|| err(lineno, "`escape` needs a mode"))?
                {
                    "none" => escape = Escape::None,
                    "backslash" => {
                        escape = Escape::Backslash(parse_byte(words.next(), lineno, "escape byte")?)
                    }
                    other => {
                        return Err(err(
                            lineno,
                            format!("unknown escape mode `{other}` (expected none or backslash)"),
                        ));
                    }
                }
            }
            "nesting" => {
                let value = words
                    .next()
                    .ok_or_else(|| err(lineno, "`nesting` needs pairs like 7b:7d"))?;
                for pair in value.split(',') {
                    let (open, close) = pair
                        .split_once(':')
                        .ok_or_else(|| err(lineno, format!("bad nesting pair `{pair}`")))?;
                    nesting.push((parse_hex(open, lineno)?, parse_hex(close, lineno)?));
                }
            }
            "lines-per-record" => {
                lines_per_record = words
                    .next()
                    .ok_or_else(|| err(lineno, "`lines-per-record` needs a count"))?
                    .parse()
                    .map_err(|_| err(lineno, "`lines-per-record` must be a positive integer"))?;
                if lines_per_record == 0 {
                    return Err(err(lineno, "`lines-per-record` must be at least 1"));
                }
            }
            "column" => {
                let index: usize = words
                    .next()
                    .ok_or_else(|| err(lineno, "`column` needs a field index"))?
                    .parse()
                    .map_err(|_| {
                        err(lineno, "column field index must be a non-negative integer")
                    })?;
                let field = words
                    .next()
                    .ok_or_else(|| err(lineno, "`column` needs a name"))?;
                let ty = match words
                    .next()
                    .ok_or_else(|| err(lineno, "`column` needs a type"))?
                {
                    "i64" => ColumnType::I64,
                    "f64" => ColumnType::F64,
                    "string" => ColumnType::Str,
                    "bytes" => ColumnType::Bytes,
                    other => {
                        return Err(err(lineno, format!("unknown column type `{other}`")));
                    }
                };
                let mut info_key = None;
                for extra in words {
                    match extra.split_once('=') {
                        Some(("info", key)) => info_key = Some(key.to_string()),
                        _ => {
                            return Err(err(lineno, format!("unknown column attribute `{extra}`")));
                        }
                    }
                }
                columns.push(Column {
                    index,
                    name: Some(field.to_string()),
                    ty,
                    info_key,
                });
            }
            "frame" => {
                let mut header = 0usize;
                let mut length_at = 0usize;
                let mut width = Width::U32;
                let mut endian = Endian::Le;
                let mut counts = Counts::Total;
                let mut adjust = 0i64;
                let mut trailer = 0usize;
                let mut magic = None;
                let mut skip_empty = false;
                for field in words {
                    let (key, value) = field.split_once('=').ok_or_else(|| {
                        err(lineno, format!("`frame` field `{field}` needs key=value"))
                    })?;
                    let num = |what: &str| -> Result<usize, IrError> {
                        value
                            .parse()
                            .map_err(|_| err(lineno, format!("`{what}` must be a number")))
                    };
                    match key {
                        "header" => header = num("header")?,
                        "length-at" => length_at = num("length-at")?,
                        "trailer" => trailer = num("trailer")?,
                        "adjust" => {
                            adjust = value
                                .parse()
                                .map_err(|_| err(lineno, "`adjust` must be a signed integer"))?
                        }
                        "width" => {
                            width = match value {
                                "u8" => Width::U8,
                                "u16" => Width::U16,
                                "u32" => Width::U32,
                                "u64" => Width::U64,
                                "varint" => Width::Varint,
                                other => {
                                    return Err(err(
                                        lineno,
                                        format!("unknown frame width `{other}`"),
                                    ));
                                }
                            }
                        }
                        "endian" => {
                            endian = match value {
                                "le" => Endian::Le,
                                "be" => Endian::Be,
                                other => {
                                    return Err(err(
                                        lineno,
                                        format!("unknown endianness `{other}`"),
                                    ));
                                }
                            }
                        }
                        "counts" => {
                            counts = match value {
                                "total" => Counts::Total,
                                "payload" => Counts::Payload,
                                other => {
                                    return Err(err(
                                        lineno,
                                        format!("`counts` must be total or payload, got `{other}`"),
                                    ));
                                }
                            }
                        }
                        "magic" => {
                            let (offset, bytes) = value.split_once(':').ok_or_else(|| {
                                err(lineno, "`magic` looks like <offset>:<hex bytes>")
                            })?;
                            let offset: usize = offset
                                .parse()
                                .map_err(|_| err(lineno, "`magic` offset must be a number"))?;
                            magic = Some((offset, parse_byte_list(bytes, lineno)?));
                        }
                        "skip-empty" => skip_empty = value == "true",
                        other => {
                            return Err(err(lineno, format!("unknown `frame` field `{other}`")));
                        }
                    }
                }
                framing = Some(Framing {
                    header,
                    length_at,
                    width,
                    endian,
                    counts,
                    adjust,
                    trailer,
                    magic,
                    skip_empty,
                });
            }
            "output" => output = Some(parse_ref(words.next(), next_node, lineno)?),
            "terminators" => terminators = Some(parse_ref(words.next(), next_node, lineno)?),
            "nest-open" => nest_open = Some(parse_ref(words.next(), next_node, lineno)?),
            "nest-close" => nest_close = Some(parse_ref(words.next(), next_node, lineno)?),
            other => {
                return Err(err(lineno, format!("unknown directive `{other}`")));
            }
        }
    }

    if !version_seen {
        return Err(IrError("empty input: expected a `falx-ir` header".into()));
    }
    let name = name.ok_or_else(|| IrError("missing `format` directive".into()))?;
    let structural = structural.ok_or_else(|| IrError("missing `structural` directive".into()))?;
    let output = output.ok_or_else(|| IrError("missing `output` directive".into()))?;
    let terminators =
        terminators.ok_or_else(|| IrError("missing `terminators` directive".into()))?;
    let nest = match (nest_open, nest_close) {
        (Some(open), Some(close)) => Some((open, close)),
        (None, None) => None,
        _ => {
            return Err(IrError(
                "`nest-open` and `nest-close` must appear together".into(),
            ));
        }
    };
    graph.set_output(output);

    Ok(Module {
        name,
        dialect: Dialect {
            structural,
            quote,
            escape,
            comment,
            nesting,
            lines_per_record,
        },
        columns,
        graph,
        terminators,
        nest,
        framing,
    })
}

/// Append one node to `graph`, validating that every operand was already
/// defined (`< defining`).
fn parse_op(
    graph: &mut Graph,
    opcode: &str,
    operands: &[&str],
    defining: usize,
    lineno: usize,
) -> Result<NodeId, IrError> {
    let arity = |want: usize| -> Result<(), IrError> {
        if operands.len() == want {
            Ok(())
        } else {
            Err(err(
                lineno,
                format!("`{opcode}` takes {want} operand(s), got {}", operands.len()),
            ))
        }
    };
    let operand = |i: usize| -> Result<NodeId, IrError> {
        parse_ref(operands.get(i).copied(), defining, lineno)
    };
    Ok(match opcode {
        "class" => {
            arity(1)?;
            graph.class(parse_class(operands[0], lineno)?)
        }
        "const" => {
            arity(1)?;
            let text = operands[0];
            let value = match text.strip_prefix("0x").or_else(|| text.strip_prefix("0X")) {
                Some(hex) => u64::from_str_radix(&hex.replace('_', ""), 16),
                None => text.replace('_', "").parse::<u64>(),
            }
            .map_err(|_| err(lineno, format!("bad constant `{text}`")))?;
            graph.constant(value)
        }
        "not" => {
            arity(1)?;
            graph.not(operand(0)?)
        }
        "and" => {
            arity(2)?;
            graph.and(operand(0)?, operand(1)?)
        }
        "or" => {
            arity(2)?;
            graph.or(operand(0)?, operand(1)?)
        }
        "xor" => {
            arity(2)?;
            graph.xor(operand(0)?, operand(1)?)
        }
        "shl1" => {
            arity(1)?;
            graph.shift_left1(operand(0)?)
        }
        "shl1-seeded" => {
            arity(1)?;
            graph.shift_left1_seeded(operand(0)?)
        }
        "regions" => {
            arity(3)?;
            graph.regions(operand(0)?, operand(1)?, operand(2)?)
        }
        "prefix-xor" => {
            arity(1)?;
            graph.prefix_xor(operand(0)?)
        }
        "add" => {
            arity(2)?;
            graph.add(operand(0)?, operand(1)?)
        }
        other => return Err(err(lineno, format!("unknown opcode `{other}`"))),
    })
}

/// Parse a `%k` node reference, rejecting forward and dangling references.
fn parse_ref(token: Option<&str>, limit: usize, lineno: usize) -> Result<NodeId, IrError> {
    let token = token.ok_or_else(|| err(lineno, "expected a node reference"))?;
    let rest = token
        .strip_prefix('%')
        .ok_or_else(|| err(lineno, format!("expected `%k`, got `{token}`")))?;
    let index: usize = rest
        .parse()
        .map_err(|_| err(lineno, format!("bad node reference `{token}`")))?;
    if index >= limit {
        return Err(err(
            lineno,
            format!("`{token}` refers to a node that is not defined yet"),
        ));
    }
    Ok(NodeId::from_index(index))
}

fn parse_class(text: &str, lineno: usize) -> Result<CharClass, IrError> {
    if text == "-" {
        return Ok(CharClass::empty());
    }
    let mut bytes = Vec::new();
    for part in text.split(',') {
        match part.split_once('-') {
            Some((lo, hi)) => {
                let (lo, hi) = (parse_hex(lo, lineno)?, parse_hex(hi, lineno)?);
                if lo > hi {
                    return Err(err(lineno, format!("descending byte range `{part}`")));
                }
                bytes.extend(lo..=hi);
            }
            None => bytes.push(parse_hex(part, lineno)?),
        }
    }
    Ok(CharClass::from_bytes(&bytes))
}

fn parse_byte_list(text: &str, lineno: usize) -> Result<Vec<u8>, IrError> {
    if text == "-" {
        return Ok(Vec::new());
    }
    text.split(',').map(|p| parse_hex(p, lineno)).collect()
}

fn parse_byte(token: Option<&str>, lineno: usize, what: &str) -> Result<u8, IrError> {
    let token = token.ok_or_else(|| err(lineno, format!("`{what}` needs a hex byte")))?;
    parse_hex(token, lineno)
}

fn parse_hex(text: &str, lineno: usize) -> Result<u8, IrError> {
    u8::from_str_radix(text, 16)
        .map_err(|_| err(lineno, format!("`{text}` is not a two-digit hex byte")))
}
