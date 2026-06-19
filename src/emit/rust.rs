//! The Rust [`Backend`]: renders the [`ast`](super::ast) to idiomatic Rust.

use super::ast::{BinOp, Block, Comment, Expr, Item, Stmt, Type, UnOp};
use super::{
    Backend, EmitError, PREC_POSTFIX, PREC_UNARY, Printer, infix_prec, infix_symbol, parens_if,
};

/// Renders an [`ast`](super::ast) program as Rust source.
pub struct Rust;

impl Backend for Rust {
    fn name(&self) -> &'static str {
        "rust"
    }

    fn emit(&self, items: &[Item]) -> Result<String, EmitError> {
        let mut p = Printer::default();
        for (i, it) in items.iter().enumerate() {
            if i > 0 {
                p.blank();
            }
            item(&mut p, it)?;
        }
        Ok(p.finish())
    }
}

fn ty(t: &Type) -> String {
    match t {
        Type::Name(n) => n.clone(),
        Type::Slice(inner) => format!("&[{}]", ty(inner)),
        Type::SliceMut(inner) => format!("&mut [{}]", ty(inner)),
        Type::Ref(inner) => format!("&{}", ty(inner)),
        Type::RefMut(inner) => format!("&mut {}", ty(inner)),
        Type::Unit => "()".to_string(),
        Type::Raw(s) => s.clone(),
    }
}

fn item(p: &mut Printer, it: &Item) -> Result<(), EmitError> {
    match it {
        Item::Use(path) => p.line(&format!("use {path};")),
        Item::Comment(c) => comment(p, c),
        Item::Raw(s) => {
            for ln in s.lines() {
                p.line(ln);
            }
        }
        Item::Const {
            doc,
            name,
            ty: t,
            value,
        } => {
            if let Some(d) = doc {
                p.line(&format!("/// {d}"));
            }
            p.line(&format!("const {name}: {} = {};", ty(t), expr(value, 0)?));
        }
        Item::Struct(s) => {
            for d in &s.doc {
                p.line(&format!("/// {d}"));
            }
            for a in &s.attrs {
                p.line(&format!("#[{}]", a.0));
            }
            let vis = if s.vis_pub { "pub " } else { "" };
            p.line(&format!("{vis}struct {} {{", s.name));
            p.indent();
            for f in &s.fields {
                if let Some(d) = &f.doc {
                    p.line(&format!("/// {d}"));
                }
                let fvis = if f.vis_pub { "pub " } else { "" };
                p.line(&format!("{fvis}{}: {},", f.name, ty(&f.ty)));
            }
            p.dedent();
            p.line("}");
        }
        Item::Func(func) => {
            for d in &func.doc {
                p.line(&format!("/// {d}"));
            }
            for a in &func.attrs {
                p.line(&format!("#[{}]", a.0));
            }
            let vis = if func.vis_pub { "pub " } else { "" };
            let unsafe_ = if func.is_unsafe { "unsafe " } else { "" };
            let params = func
                .params
                .iter()
                .map(|prm| format!("{}: {}", prm.name, ty(&prm.ty)))
                .collect::<Vec<_>>()
                .join(", ");
            let ret = match &func.ret {
                Some(r) => format!(" -> {}", ty(r)),
                None => String::new(),
            };
            p.line(&format!("{vis}{unsafe_}fn {}({params}){ret} {{", func.name));
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
        Comment::Line(s) => p.line(&format!("// {s}")),
        Comment::Doc(s) => p.line(&format!("/// {s}")),
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
            mutable,
            ty: t,
            init,
        } => {
            let m = if *mutable { "mut " } else { "" };
            let ann = match t {
                Some(t) => format!(": {}", ty(t)),
                None => String::new(),
            };
            p.line(&format!("let {m}{name}{ann} = {};", expr(init, 0)?));
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
                "for {var} in {}..{} {{",
                expr(start, PREC_POSTFIX)?,
                expr(end, PREC_POSTFIX)?
            ));
            p.indent();
            block(p, body)?;
            p.dedent();
            p.line("}");
        }
        Stmt::While { cond, body } => {
            p.line(&format!("while {} {{", expr(cond, 0)?));
            p.indent();
            block(p, body)?;
            p.dedent();
            p.line("}");
        }
        Stmt::If { cond, then, els } => {
            p.line(&format!("if {} {{", expr(cond, 0)?));
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

/// Render an expression. `parent` is the precedence of the surrounding context;
/// the result is parenthesized when this expression binds more loosely.
fn expr(e: &Expr, parent: u8) -> Result<String, EmitError> {
    Ok(match e {
        Expr::Path(s) => s.clone(),
        Expr::Int { value, hex } => {
            if *hex {
                format!("0x{value:x}")
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
        Expr::SliceLen(inner) => format!("{}.len()", expr(inner, PREC_POSTFIX)?),
        Expr::Cast { expr: inner, ty: t } => {
            let s = format!("{} as {}", expr(inner, PREC_UNARY)?, ty(t));
            parens_if(s, PREC_UNARY, parent)
        }
        Expr::Unary { op, expr: inner } => {
            let sym = match op {
                UnOp::Not => "!",
                UnOp::Neg => "-",
            };
            let s = format!("{sym}{}", expr(inner, PREC_UNARY)?);
            parens_if(s, PREC_UNARY, parent)
        }
        // Wrapping arithmetic is a method call in Rust: postfix-precedence.
        Expr::Binary {
            op: op @ (BinOp::WrapAdd | BinOp::WrapMul),
            lhs,
            rhs,
        } => {
            let method = if *op == BinOp::WrapMul {
                "wrapping_mul"
            } else {
                "wrapping_add"
            };
            let s = format!("{}.{method}({})", expr(lhs, PREC_POSTFIX)?, expr(rhs, 0)?);
            parens_if(s, PREC_POSTFIX, parent)
        }
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
