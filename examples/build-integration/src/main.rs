// The generated parser is produced by build.rs from spec.toml at compile
// time; this crate has no runtime dependency on falx.
mod parser {
    include!(concat!(env!("OUT_DIR"), "/parser.rs"));
}

fn main() {
    let csv_data = b"name,age\n\"Smith, \"\"Bob\"\"\",30\n";
    println!("Input CSV:\n{}", String::from_utf8_lossy(csv_data));

    let parsed = parser::parse(csv_data);
    for (i, record) in parsed.records().enumerate() {
        println!("Record {i}:");
        for field in record.fields() {
            // fields() yields Cow<[u8]>: quotes stripped, "" unescaped,
            // borrowing unless an escape forced a copy.
            println!("  {:?}", String::from_utf8_lossy(&field));
        }
    }

    let second = parsed.records().nth(1).expect("two records");
    assert_eq!(
        second.field(0).expect("field present").as_ref(),
        b"Smith, \"Bob\"",
        "quoted field should come out unquoted and unescaped"
    );
    println!("\nAssertion passed: quoted field unescaped to `Smith, \"Bob\"`.");
}
