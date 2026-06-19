//! A small, backend-agnostic AST for the *code falx generates*.
//!
//! This is deliberately NOT [`crate::ir`] — `ir::Graph` is the bitstream
//! algebra falx optimizes; this AST models the *target program* (items,
//! functions, statements, expressions) that a backend renders to source text.
//!
//! Two properties motivate a hand-rolled AST over a `quote!` `TokenStream`:
//!
//! 1. **Comments are first-class.** [`Comment`] nodes survive to the output, so
//!    generated kernels stay the readable, auditable artifacts they are today.
//!    A `TokenStream` cannot carry `//` line comments at all.
//! 2. **One tree, many backends.** The same AST lowers to idiomatic Rust *or*
//!    CUDA-C (see [`crate::emit::rust`] / [`crate::emit::c`]). Backend-specific
//!    shape (a `&[u8]` slice vs. a `const uint8_t*, size_t` pair; a
//!    `wrapping_mul` method vs. a plain `*`) is decided by the renderer, not
//!    baked into the tree.
//!
//! Every level has a `Raw` escape hatch, so the long tail of constructs we do
//! not model yet (SIMD intrinsics, an exotic statement) can still be emitted as
//! verbatim text while the surrounding structure stays typed.

/// A documentation or inline comment, preserved through to the rendered source.
#[derive(Clone, Debug)]
pub enum Comment {
    /// `// text` (Rust) / `// text` (C).
    Line(String),
    /// `/// text` doc comment (Rust) / `/** text */` (C).
    Doc(String),
}

/// A type reference in the target program.
#[derive(Clone, Debug)]
pub enum Type {
    /// A named type. Scalar names (`u8`, `u64`, `usize`, `f64`, `bool`, …) are
    /// mapped to the backend's spelling (`uint8_t`, `size_t`, `double`, …);
    /// unknown names pass through unchanged.
    Name(String),
    /// `&[T]` — in C this expands a parameter into `const T*, size_t len`.
    Slice(Box<Type>),
    /// `&mut [T]` — in C, `T*, size_t len`.
    SliceMut(Box<Type>),
    /// `&T` / `const T*`.
    Ref(Box<Type>),
    /// `&mut T` / `T*`.
    RefMut(Box<Type>),
    /// `()` / `void`.
    Unit,
    /// Verbatim, backend-specific type text.
    Raw(String),
}

impl Type {
    /// A named type, e.g. `Type::name("u64")`.
    pub fn name(n: impl Into<String>) -> Type {
        Type::Name(n.into())
    }
    /// `&[inner]`.
    pub fn slice(inner: Type) -> Type {
        Type::Slice(Box::new(inner))
    }
    /// `&mut [inner]`.
    pub fn slice_mut(inner: Type) -> Type {
        Type::SliceMut(Box::new(inner))
    }
    /// `&inner`.
    pub fn reference(inner: Type) -> Type {
        Type::Ref(Box::new(inner))
    }
    /// `&mut inner`.
    pub fn reference_mut(inner: Type) -> Type {
        Type::RefMut(Box::new(inner))
    }
}

/// A binary operator. A few carry *semantics* (not just a symbol) so each
/// backend can render them correctly: `WrapMul` is `a.wrapping_mul(b)` in Rust
/// but a plain `a * b` in C (unsigned multiplication wraps by definition).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    /// Wrapping add: Rust `a.wrapping_add(b)`, C `a + b`.
    WrapAdd,
    /// Wrapping multiply: Rust `a.wrapping_mul(b)`, C `a * b`.
    WrapMul,
    BitXor,
    BitOr,
    BitAnd,
    Shl,
    Shr,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    /// Short-circuiting logical `&&`.
    AndAnd,
    /// Short-circuiting logical `||`.
    OrOr,
}

/// A unary operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnOp {
    /// `!x` (Rust bitwise/logical not) / `~x` or `!x` (C, by operand type).
    Not,
    /// `-x`.
    Neg,
}

/// An expression.
#[derive(Clone, Debug)]
pub enum Expr {
    /// An identifier or path: `x`, `self.0`, `foo::bar`.
    Path(String),
    /// An unsigned integer literal; `hex` selects `0x…` formatting.
    Int { value: u64, hex: bool },
    /// A boolean literal.
    Bool(bool),
    /// `lhs <op> rhs` (rendering depends on `op` and backend).
    Binary {
        op: BinOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    /// `<op>expr`.
    Unary { op: UnOp, expr: Box<Expr> },
    /// `base[index]`.
    Index { base: Box<Expr>, index: Box<Expr> },
    /// `base.field`.
    Field { base: Box<Expr>, field: String },
    /// `func(args…)`.
    Call { func: Box<Expr>, args: Vec<Expr> },
    /// `expr as ty` (Rust) / `(ty)expr` (C).
    Cast { expr: Box<Expr>, ty: Type },
    /// Length of a slice value: Rust `expr.len()`; C `<name>_len`, using the
    /// companion length parameter a slice expands to (so `expr` must be a path).
    SliceLen(Box<Expr>),
    /// Verbatim, backend-specific expression text.
    Raw(String),
}

impl Expr {
    /// An identifier/path expression.
    pub fn path(p: impl Into<String>) -> Expr {
        Expr::Path(p.into())
    }
    /// A decimal integer literal.
    pub fn int(value: u64) -> Expr {
        Expr::Int { value, hex: false }
    }
    /// A hexadecimal integer literal.
    pub fn hex(value: u64) -> Expr {
        Expr::Int { value, hex: true }
    }
    /// Verbatim expression text (escape hatch).
    pub fn raw(s: impl Into<String>) -> Expr {
        Expr::Raw(s.into())
    }
    /// `lhs <op> rhs`.
    pub fn binary(op: BinOp, lhs: Expr, rhs: Expr) -> Expr {
        Expr::Binary {
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
    }
    /// `base[index]`.
    pub fn index(base: Expr, index: Expr) -> Expr {
        Expr::Index {
            base: Box::new(base),
            index: Box::new(index),
        }
    }
    /// `expr as ty`.
    pub fn cast(expr: Expr, ty: Type) -> Expr {
        Expr::Cast {
            expr: Box::new(expr),
            ty,
        }
    }
    /// `func(args…)`.
    pub fn call(func: Expr, args: Vec<Expr>) -> Expr {
        Expr::Call {
            func: Box::new(func),
            args,
        }
    }
    /// `<op>expr`.
    pub fn unary(op: UnOp, expr: Expr) -> Expr {
        Expr::Unary {
            op,
            expr: Box::new(expr),
        }
    }
    /// Length of a slice path.
    pub fn slice_len(expr: Expr) -> Expr {
        Expr::SliceLen(Box::new(expr))
    }
}

/// A statement inside a function body.
#[derive(Clone, Debug)]
pub enum Stmt {
    /// `let [mut] name[: ty] = init;`. A type is required for the C backend
    /// (C has no inference); the Rust backend prints it when present.
    Let {
        name: String,
        mutable: bool,
        ty: Option<Type>,
        init: Expr,
    },
    /// `target = value;` or, with `op`, a compound assignment `target op= value;`.
    Assign {
        target: Expr,
        op: Option<BinOp>,
        value: Expr,
    },
    /// A bare expression statement.
    Expr(Expr),
    /// `return [value];`.
    Return(Option<Expr>),
    /// A counting loop over `start..end`. The C backend uses a `size_t` counter.
    ForRange {
        var: String,
        start: Expr,
        end: Expr,
        body: Block,
    },
    /// `while cond { body }`.
    While { cond: Expr, body: Block },
    /// `if cond { then } [else { els }]`.
    If {
        cond: Expr,
        then: Block,
        els: Option<Block>,
    },
    /// A comment occupying its own line.
    Comment(Comment),
    /// Verbatim statement text (escape hatch); rendered as-is on its own line.
    Raw(String),
}

impl Stmt {
    /// `let [mut] name: ty = init;`.
    pub fn let_(name: impl Into<String>, mutable: bool, ty: Type, init: Expr) -> Stmt {
        Stmt::Let {
            name: name.into(),
            mutable,
            ty: Some(ty),
            init,
        }
    }
    /// `target op= value;`.
    pub fn assign_op(target: Expr, op: BinOp, value: Expr) -> Stmt {
        Stmt::Assign {
            target,
            op: Some(op),
            value,
        }
    }
    /// `target = value;`.
    pub fn assign(target: Expr, value: Expr) -> Stmt {
        Stmt::Assign {
            target,
            op: None,
            value,
        }
    }
    /// `return value;`.
    pub fn ret(value: Expr) -> Stmt {
        Stmt::Return(Some(value))
    }
    /// `for var in start..end { body }`.
    pub fn for_range(var: impl Into<String>, start: Expr, end: Expr, body: Vec<Stmt>) -> Stmt {
        Stmt::ForRange {
            var: var.into(),
            start,
            end,
            body: Block(body),
        }
    }
    /// A `// text` line comment.
    pub fn line_comment(text: impl Into<String>) -> Stmt {
        Stmt::Comment(Comment::Line(text.into()))
    }
}

/// A brace-delimited sequence of statements.
#[derive(Clone, Debug, Default)]
pub struct Block(pub Vec<Stmt>);

/// A Rust attribute / its C analogue, e.g. `inline` or
/// `target_feature(enable = "avx2")`. Rust renders `#[..]`; the C backend
/// renders the ones it understands (`inline`) and drops the rest.
#[derive(Clone, Debug)]
pub struct Attr(pub String);

/// A function parameter.
#[derive(Clone, Debug)]
pub struct Param {
    pub name: String,
    pub ty: Type,
}

/// A free function.
#[derive(Clone, Debug)]
pub struct Func {
    pub doc: Vec<String>,
    pub attrs: Vec<Attr>,
    pub vis_pub: bool,
    pub is_unsafe: bool,
    pub name: String,
    pub params: Vec<Param>,
    pub ret: Option<Type>,
    pub body: Block,
}

impl Func {
    /// An empty private function named `name`.
    pub fn new(name: impl Into<String>) -> Func {
        Func {
            doc: Vec::new(),
            attrs: Vec::new(),
            vis_pub: false,
            is_unsafe: false,
            name: name.into(),
            params: Vec::new(),
            ret: None,
            body: Block(Vec::new()),
        }
    }
    /// Mark the function `pub`.
    pub fn public(mut self) -> Func {
        self.vis_pub = true;
        self
    }
    /// Append a doc-comment line.
    pub fn doc(mut self, line: impl Into<String>) -> Func {
        self.doc.push(line.into());
        self
    }
    /// Append an attribute (without the `#[ ]`).
    pub fn attr(mut self, a: impl Into<String>) -> Func {
        self.attrs.push(Attr(a.into()));
        self
    }
    /// Append a parameter.
    pub fn param(mut self, name: impl Into<String>, ty: Type) -> Func {
        self.params.push(Param {
            name: name.into(),
            ty,
        });
        self
    }
    /// Set the return type.
    pub fn ret(mut self, ty: Type) -> Func {
        self.ret = Some(ty);
        self
    }
    /// Set the body.
    pub fn body(mut self, stmts: Vec<Stmt>) -> Func {
        self.body = Block(stmts);
        self
    }
}

/// A struct field.
#[derive(Clone, Debug)]
pub struct Field {
    pub doc: Option<String>,
    pub name: String,
    pub ty: Type,
    pub vis_pub: bool,
}

/// A struct definition.
#[derive(Clone, Debug)]
pub struct Struct {
    pub doc: Vec<String>,
    pub attrs: Vec<Attr>,
    pub vis_pub: bool,
    pub name: String,
    pub fields: Vec<Field>,
}

/// A top-level item. The large variants are boxed so the enum stays small.
#[derive(Clone, Debug)]
pub enum Item {
    Func(Box<Func>),
    Struct(Box<Struct>),
    Const {
        doc: Option<String>,
        name: String,
        ty: Type,
        value: Expr,
    },
    /// A `use` path (Rust). The C backend treats this as an `#include` hint or
    /// drops it.
    Use(String),
    /// A free-standing comment.
    Comment(Comment),
    /// Verbatim item text (escape hatch).
    Raw(String),
}

impl From<Func> for Item {
    fn from(f: Func) -> Item {
        Item::Func(Box::new(f))
    }
}

impl From<Struct> for Item {
    fn from(s: Struct) -> Item {
        Item::Struct(Box::new(s))
    }
}
