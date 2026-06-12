//! TOML spec parsing for delimited format dialects.
//!
//! This module provides utilities for parsing declarative format specifications
//! in TOML format into [`Spec`] values that can be consumed by code generation.

use crate::codegen::{Column, ColumnType};
use crate::formats::{Dialect, Escape};

/// A parsed format specification from TOML.
#[derive(Debug)]
pub struct Spec {
    /// The name of the format (e.g., "csv").
    pub name: String,
    /// The parsed dialect with structural bytes, quote byte, and escape style.
    pub dialect: Dialect,
    /// Typed columns to project (empty = no columnar API generated).
    pub columns: Vec<Column>,
}

/// Parse a TOML spec string into a [`Spec`].
///
/// # Required fields
/// - `name` (string): The format name.
/// - `structural` (array of strings): Single-byte structural characters.
///
/// # Optional fields
/// - `quote` (string): Single-byte quote character (default: none).
/// - `escape` (string): Escape style: "none", "doubled", or "backslash" (default: "none").
/// - `escape_char` (string): Single-byte escape character for backslash mode (default: "\\").
/// - `[[columns]]` (array of tables): Typed columns to project. Each entry
///   has `index` (integer, zero-based field index), `type` (string: "i64",
///   "f64", or "bytes"), and optional `name` (string: generated field name,
///   default `c{index}`).
///
/// # Errors
/// Returns descriptive error messages for missing required fields, type mismatches,
/// or invalid values.
pub fn parse(toml_text: &str) -> Result<Spec, String> {
    let parsed: toml::Table = toml_text
        .parse()
        .map_err(|e| format!("Failed to parse TOML: {}", e))?;

    // Extract `name` (required, string).
    let name = parsed
        .get("name")
        .and_then(|v| v.as_str())
        .ok_or("Missing required field: 'name' (must be a string)")?
        .to_string();

    // Extract `structural` (required, array of strings).
    let structural_arr = parsed
        .get("structural")
        .and_then(|v| v.as_array())
        .ok_or("Missing required field: 'structural' (must be an array of strings)")?;

    let mut structural = Vec::new();
    for (idx, item) in structural_arr.iter().enumerate() {
        let s = item
            .as_str()
            .ok_or(format!(
                "structural[{}]: expected string, got {}",
                idx,
                value_type_name(item)
            ))?;
        let byte = parse_string_as_byte(s).map_err(|e| {
            format!("structural[{}]: {}", idx, e)
        })?;
        structural.push(byte);
    }

    // Extract `record_terminator` (optional, string, single byte; default "\n").
    let record_terminator = match parsed.get("record_terminator") {
        Some(value) => {
            let s = value
                .as_str()
                .ok_or("'record_terminator' must be a string")?;
            parse_string_as_byte(s)
                .map_err(|e| format!("'record_terminator': {}", e))?
        }
        None => b'\n',
    };

    // Extract `quote` (optional, string, single byte).
    let quote = match parsed.get("quote") {
        Some(value) => {
            let s = value.as_str().ok_or("'quote' must be a string")?;
            Some(parse_string_as_byte(s).map_err(|e| format!("'quote': {}", e))?)
        }
        None => None,
    };

    // Extract `escape` (optional, string; default "none"; valid: "none", "doubled", "backslash").
    let escape_str = parsed
        .get("escape")
        .and_then(|v| v.as_str())
        .unwrap_or("none");

    // Extract `escape_char` (optional, string, single byte; default "\\").
    let escape_char_str = parsed
        .get("escape_char")
        .and_then(|v| v.as_str())
        .unwrap_or("\\");
    let escape_char = parse_string_as_byte(escape_char_str)
        .map_err(|e| format!("'escape_char': {}", e))?;

    let escape = match escape_str {
        "none" | "doubled" => Escape::None,
        "backslash" => Escape::Backslash(escape_char),
        other => {
            return Err(format!(
                "Invalid 'escape' value: '{}' (must be 'none', 'doubled', or 'backslash')",
                other
            ))
        }
    };

    let dialect = Dialect {
        structural,
        record_terminator,
        quote,
        escape,
    };

    // Extract `[[columns]]` (optional, array of tables).
    let mut columns = Vec::new();
    if let Some(value) = parsed.get("columns") {
        let arr = value
            .as_array()
            .ok_or("'columns' must be an array of tables ([[columns]])")?;
        for (i, item) in arr.iter().enumerate() {
            let table = item.as_table().ok_or(format!(
                "columns[{}]: expected a table, got {}",
                i,
                value_type_name(item)
            ))?;
            let index = table
                .get("index")
                .and_then(|v| v.as_integer())
                .ok_or(format!("columns[{i}]: missing required field 'index' (integer)"))?;
            let index = usize::try_from(index)
                .map_err(|_| format!("columns[{i}]: 'index' must be non-negative"))?;
            let ty = match table.get("type").and_then(|v| v.as_str()) {
                Some("i64") => ColumnType::I64,
                Some("f64") => ColumnType::F64,
                Some("bytes") => ColumnType::Bytes,
                Some(other) => {
                    return Err(format!(
                        "columns[{i}]: invalid 'type' value '{other}' (must be 'i64', 'f64', or 'bytes')"
                    ))
                }
                None => {
                    return Err(format!(
                        "columns[{i}]: missing required field 'type' (string)"
                    ))
                }
            };
            let name = match table.get("name") {
                Some(v) => Some(
                    v.as_str()
                        .ok_or(format!("columns[{i}]: 'name' must be a string"))?
                        .to_string(),
                ),
                None => None,
            };
            columns.push(Column { index, name, ty });
        }
    }

    Ok(Spec { name, dialect, columns })
}

/// Parse a TOML string value as a single byte using Rust's escape sequences.
fn parse_string_as_byte(s: &str) -> Result<u8, String> {
    let bytes = s.as_bytes();

    // Handle escape sequences manually.
    let unescaped = if s.starts_with('\\') && s.len() == 2 {
        match &s[1..2] {
            "n" => b'\n',
            "t" => b'\t',
            "r" => b'\r',
            "\\" => b'\\',
            "\"" => b'"',
            _ => return Err(format!("Unknown escape sequence in string: {}", s)),
        }
    } else if bytes.len() == 1 {
        bytes[0]
    } else {
        return Err(format!(
            "String '{}' is not a single byte (interpreted as {} bytes)",
            s, bytes.len()
        ));
    };

    Ok(unescaped)
}

/// Return a human-readable name for a TOML value type.
fn value_type_name(v: &toml::Value) -> &'static str {
    match v {
        toml::Value::String(_) => "string",
        toml::Value::Integer(_) => "integer",
        toml::Value::Float(_) => "float",
        toml::Value::Boolean(_) => "boolean",
        toml::Value::Datetime(_) => "datetime",
        toml::Value::Array(_) => "array",
        toml::Value::Table(_) => "table",
    }
}
