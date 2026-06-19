//! An experimental **typed-AST code emitter**, living alongside the existing
//! string-template generator in [`crate::codegen`] (which is untouched).
//!
//! The string templates in `codegen` produce Rust source directly. That works,
//! but it interleaves logic with hundreds of lines of `r#"…"#` and is single-
//! target. This module explores the alternative the way most compilers emit
//! code: build a small typed AST ([`ast`]), then render it with a [`Backend`].
//!
//! The payoff is multi-backend output from one tree. The same AST renders to
//! idiomatic Rust ([`rust::Rust`]) or CUDA-C ([`c::C`]); the divergences
//! (`&[u8]` vs. `const uint8_t*, size_t`; `wrapping_mul` vs. `*`) live in the
//! renderers, not the tree. Comments are first-class AST nodes, so generated
//! code stays readable — something a `quote!` `TokenStream` cannot do.
//!
//! ```
//! use falx::emit::{ast::*, emit_rust, emit_c};
//!
//! // fn add(a: u64, b: u64) -> u64 { return a + b; }
//! let f = Func::new("add")
//!     .public()
//!     .param("a", Type::name("u64"))
//!     .param("b", Type::name("u64"))
//!     .ret(Type::name("u64"))
//!     .body(vec![Stmt::ret(Expr::binary(
//!         BinOp::Add,
//!         Expr::path("a"),
//!         Expr::path("b"),
//!     ))]);
//! let items = [Item::from(f)];
//! assert!(emit_rust(&items).unwrap().contains("pub fn add(a: u64, b: u64) -> u64"));
//! assert!(emit_c(&items).unwrap().contains("uint64_t add(uint64_t a, uint64_t b)"));
//! ```

pub mod ast;
pub mod c;
pub mod lower;
pub mod rust;

use ast::{BinOp, Item};

/// An error produced while rendering an AST to a particular backend, e.g. a
/// construct a backend cannot express (a Rust `impl` has no C analogue).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EmitError(pub String);

impl std::fmt::Display for EmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "emit error: {}", self.0)
    }
}

impl std::error::Error for EmitError {}

/// A code generator backend: renders a sequence of top-level [`Item`]s to
/// source text in some target language.
pub trait Backend {
    /// Short identifier for the target language, e.g. `"rust"` or `"c"`.
    fn name(&self) -> &'static str;
    /// Render a whole program to source text.
    fn emit(&self, items: &[Item]) -> Result<String, EmitError>;
}

/// Render `items` as Rust source.
pub fn emit_rust(items: &[Item]) -> Result<String, EmitError> {
    rust::Rust.emit(items)
}

/// Render `items` as CUDA-C source.
pub fn emit_c(items: &[Item]) -> Result<String, EmitError> {
    c::C.emit(items)
}

// --- Shared rendering helpers, used by both backends -----------------------

/// Precedence of postfix forms (call, index, field, method) — nothing ever
/// needs parenthesizing inside them.
pub(crate) const PREC_POSTFIX: u8 = 12;
/// Precedence of unary operators and casts.
pub(crate) const PREC_UNARY: u8 = 11;

/// Binding strength of a binary operator when rendered infix. Higher binds
/// tighter; used to decide where parentheses are required.
pub(crate) fn infix_prec(op: BinOp) -> u8 {
    use BinOp::*;
    match op {
        Mul | WrapMul => 10,
        Add | Sub | WrapAdd => 9,
        Shl | Shr => 8,
        BitAnd => 7,
        BitXor => 6,
        BitOr => 5,
        Eq | Ne | Lt | Le | Gt | Ge => 4,
        AndAnd => 3,
        OrOr => 2,
    }
}

/// The infix operator symbol shared by both target languages. (The Rust
/// backend renders `WrapAdd`/`WrapMul` as method calls instead and never asks
/// for their symbol; the C backend uses `+` / `*`.)
pub(crate) fn infix_symbol(op: BinOp) -> &'static str {
    use BinOp::*;
    match op {
        Add | WrapAdd => "+",
        Sub => "-",
        Mul | WrapMul => "*",
        BitXor => "^",
        BitOr => "|",
        BitAnd => "&",
        Shl => "<<",
        Shr => ">>",
        Eq => "==",
        Ne => "!=",
        Lt => "<",
        Le => "<=",
        Gt => ">",
        Ge => ">=",
        AndAnd => "&&",
        OrOr => "||",
    }
}

/// Wrap `s` in parentheses when an expression of strength `mine` appears in a
/// position demanding at least `needed`.
pub(crate) fn parens_if(s: String, mine: u8, needed: u8) -> String {
    if mine < needed { format!("({s})") } else { s }
}

/// An indentation-aware source-text builder shared by the backends.
#[derive(Default)]
pub(crate) struct Printer {
    buf: String,
    indent: usize,
}

impl Printer {
    /// Emit one indented line.
    pub(crate) fn line(&mut self, s: &str) {
        for _ in 0..self.indent {
            self.buf.push_str("    ");
        }
        self.buf.push_str(s);
        self.buf.push('\n');
    }

    /// Emit a blank separator line.
    pub(crate) fn blank(&mut self) {
        self.buf.push('\n');
    }

    /// Increase the indentation level by one.
    pub(crate) fn indent(&mut self) {
        self.indent += 1;
    }

    /// Decrease the indentation level by one.
    pub(crate) fn dedent(&mut self) {
        self.indent = self.indent.saturating_sub(1);
    }

    /// Consume the printer and return the accumulated source.
    pub(crate) fn finish(self) -> String {
        self.buf
    }
}
