//! The CUDA-C [`Backend`]: renders the [`ast`](super::ast) to procedural C.
//!
//! This is the proof that one AST drives many targets. It renders the
//! procedural subset that GPU kernels actually use — free functions over
//! scalars and pointers, structs, control flow — and maps the Rust-flavored
//! nodes to their C spellings:
//!
//! * scalar names: `u64` → `uint64_t`, `usize` → `size_t`, `f64` → `double`, …
//! * a `&[T]` parameter expands to `const T* name, size_t name_len`;
//!   [`Expr::SliceLen`] of that parameter renders as `name_len`.
//! * wrapping arithmetic (`WrapAdd`/`WrapMul`) is plain `+` / `*` (unsigned C
//!   arithmetic wraps by definition).
//!
//! Constructs with no C analogue (a slice used as a non-parameter type, a `let`
//! without a type, a `SliceLen` of a non-parameter) return an [`EmitError`]
//! rather than emitting something subtly wrong.

use super::ast::{Block, Comment, Expr, Item, Stmt, Type, UnOp};
use super::{
    Backend, EmitError, PREC_POSTFIX, PREC_UNARY, Printer, infix_prec, infix_symbol, parens_if,
};

/// Renders an [`ast`](super::ast) program as CUDA-C source.
pub struct C;

impl Backend for C {
    fn name(&self) -> &'static str {
        "c"
    }

    fn emit(&self, items: &[Item]) -> Result<String, EmitError> {
        let mut p = Printer::default();
        let mut first = true;
        for it in items {
            // `use` items have no C meaning; skip without leaving a gap.
            if matches!(it, Item::Use(_)) {
                continue;
            }
            if !first {
                p.blank();
            }
            first = false;
            item(&mut p, it)?;
        }
        Ok(p.finish())
    }
}

fn scalar(name: &str) -> &str {
    match name {
        "u8" => "uint8_t",
        "u16" => "uint16_t",
        "u32" => "uint32_t",
        "u64" => "uint64_t",
        "usize" => "size_t",
        "i8" => "int8_t",
        "i16" => "int16_t",
        "i32" => "int32_t",
        "i64" => "int64_t",
        "isize" => "ptrdiff_t",
        "f32" => "float",
        "f64" => "double",
        other => other,
    }
}

fn ctype(t: &Type) -> Result<String, EmitError> {
    Ok(match t {
        Type::Name(n) => scalar(n).to_string(),
        Type::Ref(inner) => format!("const {}*", ctype(inner)?),
        Type::RefMut(inner) => format!("{}*", ctype(inner)?),
        Type::Unit => "void".to_string(),
        Type::Raw(s) => s.clone(),
        Type::Slice(_) | Type::SliceMut(_) => {
            return Err(EmitError(
                "C backend: a slice type is only valid as a function parameter".into(),
            ));
        }
    })
}

fn param(name: &str, t: &Type) -> Result<String, EmitError> {
    Ok(match t {
        Type::Slice(inner) => format!("const {}* {name}, size_t {name}_len", ctype(inner)?),
        Type::SliceMut(inner) => format!("{}* {name}, size_t {name}_len", ctype(inner)?),
        other => format!("{} {name}", ctype(other)?),
    })
}

/// C keywords/qualifiers we forward from attributes; anything else (e.g.
/// `target_feature`, which is x86-specific) is dropped.
fn c_qualifier(attr: &str) -> Option<&'static str> {
    match attr {
        "inline" => Some("inline"),
        "static" => Some("static"),
        "__global__" => Some("__global__"),
        "__device__" => Some("__device__"),
        "__host__" => Some("__host__"),
        "__forceinline__" => Some("__forceinline__"),
        _ => None,
    }
}

fn item(p: &mut Printer, it: &Item) -> Result<(), EmitError> {
    match it {
        Item::Use(_) => {} // handled (skipped) by the caller
        Item::Comment(c) => comment(p, c),
        Item::Raw(s) => {
            for ln in s.lines() {
                p.line(ln);
            }
        }
        Item::Const {
            doc,
            name,
            ty,
            value,
        } => {
            if let Some(d) = doc {
                p.line(&format!("// {d}"));
            }
            p.line(&format!(
                "static const {} {name} = {};",
                ctype(ty)?,
                expr(value, 0)?
            ));
        }
        Item::Struct(s) => {
            for d in &s.doc {
                p.line(&format!("// {d}"));
            }
            p.line(&format!("struct {} {{", s.name));
            p.indent();
            for f in &s.fields {
                if let Some(d) = &f.doc {
                    p.line(&format!("// {d}"));
                }
                p.line(&format!("{} {};", ctype(&f.ty)?, f.name));
            }
            p.dedent();
            p.line("};");
        }
        Item::Func(func) => {
            for d in &func.doc {
                p.line(&format!("// {d}"));
            }
            let mut prefix = String::new();
            for a in &func.attrs {
                if let Some(q) = c_qualifier(&a.0) {
                    prefix.push_str(q);
                    prefix.push(' ');
                }
            }
            let ret = match &func.ret {
                Some(r) => ctype(r)?,
                None => "void".to_string(),
            };
            let params = if func.params.is_empty() {
                "void".to_string()
            } else {
                let mut parts = Vec::with_capacity(func.params.len());
                for prm in &func.params {
                    parts.push(param(&prm.name, &prm.ty)?);
                }
                parts.join(", ")
            };
            p.line(&format!("{prefix}{ret} {}({params}) {{", func.name));
            p.indent();
            block(p, &func.body)?;
            p.dedent();
            p.line("}");
        }
    }
    Ok(())
}

fn comment(p: &mut Printer, c: &Comment) {
    match c {
        Comment::Line(s) | Comment::Doc(s) => p.line(&format!("// {s}")),
    }
}

fn block(p: &mut Printer, b: &Block) -> Result<(), EmitError> {
    for s in &b.0 {
        stmt(p, s)?;
    }
    Ok(())
}

fn stmt(p: &mut Printer, s: &Stmt) -> Result<(), EmitError> {
    match s {
        Stmt::Let {
            name,
            mutable: _,
            ty,
            init,
        } => {
            let t = ty.as_ref().ok_or_else(|| {
                EmitError(format!("C backend: `let {name}` needs an explicit type"))
            })?;
            p.line(&format!("{} {name} = {};", ctype(t)?, expr(init, 0)?));
        }
        Stmt::Assign { target, op, value } => {
            let opsym = op.map(infix_symbol).unwrap_or("");
            p.line(&format!(
                "{} {opsym}= {};",
                expr(target, PREC_POSTFIX)?,
                expr(value, 0)?
            ));
        }
        Stmt::Expr(e) => p.line(&format!("{};", expr(e, 0)?)),
        Stmt::Return(Some(e)) => p.line(&format!("return {};", expr(e, 0)?)),
        Stmt::Return(None) => p.line("return;"),
        Stmt::ForRange {
            var,
            start,
            end,
            body,
        } => {
            p.line(&format!(
                "for (size_t {var} = {}; {var} < {}; {var}++) {{",
                expr(start, 0)?,
                expr(end, 0)?
            ));
            p.indent();
            block(p, body)?;
            p.dedent();
            p.line("}");
        }
        Stmt::While { cond, body } => {
            p.line(&format!("while ({}) {{", expr(cond, 0)?));
            p.indent();
            block(p, body)?;
            p.dedent();
            p.line("}");
        }
        Stmt::If { cond, then, els } => {
            p.line(&format!("if ({}) {{", expr(cond, 0)?));
            p.indent();
            block(p, then)?;
            p.dedent();
            if let Some(els) = els {
                p.line("} else {");
                p.indent();
                block(p, els)?;
                p.dedent();
            }
            p.line("}");
        }
        Stmt::Comment(c) => comment(p, c),
        Stmt::Raw(s) => {
            for ln in s.lines() {
                p.line(ln);
            }
        }
    }
    Ok(())
}

fn expr(e: &Expr, parent: u8) -> Result<String, EmitError> {
    Ok(match e {
        Expr::Path(s) => s.clone(),
        Expr::Int { value, hex } => {
            if *hex {
                format!("0x{value:x}ULL")
            } else if *value > u32::MAX as u64 {
                format!("{value}ULL")
            } else {
                value.to_string()
            }
        }
        Expr::Bool(b) => b.to_string(),
        Expr::Raw(s) => s.clone(),
        Expr::Field { base, field } => format!("{}.{field}", expr(base, PREC_POSTFIX)?),
        Expr::Index { base, index } => {
            format!("{}[{}]", expr(base, PREC_POSTFIX)?, expr(index, 0)?)
        }
        Expr::Call { func, args } => {
            let mut parts = Vec::with_capacity(args.len());
            for a in args {
                parts.push(expr(a, 0)?);
            }
            format!("{}({})", expr(func, PREC_POSTFIX)?, parts.join(", "))
        }
        Expr::SliceLen(inner) => match &**inner {
            Expr::Path(name) => format!("{name}_len"),
            _ => {
                return Err(EmitError(
                    "C backend: slice length requires a slice parameter name".into(),
                ));
            }
        },
        Expr::Cast { expr: inner, ty } => {
            let s = format!("({}){}", ctype(ty)?, expr(inner, PREC_UNARY)?);
            parens_if(s, PREC_UNARY, parent)
        }
        Expr::Unary { op, expr: inner } => {
            let sym = match op {
                UnOp::Not => "~",
                UnOp::Neg => "-",
            };
            let s = format!("{sym}{}", expr(inner, PREC_UNARY)?);
            parens_if(s, PREC_UNARY, parent)
        }
        // Wrapping ops are plain infix arithmetic in C (unsigned wraps).
        Expr::Binary { op, lhs, rhs } => {
            let pr = infix_prec(*op);
            let s = format!(
                "{} {} {}",
                expr(lhs, pr)?,
                infix_symbol(*op),
                expr(rhs, pr + 1)?
            );
            parens_if(s, pr, parent)
        }
    })
}
