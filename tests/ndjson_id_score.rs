use falx::kernels::ndjson;

#[test]
fn ndjson_id_score_sum_matches_expected_serial_and_parallel() {
    let data = br#"{"id":10,"name":"a","tags":["x","y"],"nested":{"score":7,"ok":true},"note":"plain"}
{"id":-3,"name":"quoted \" id\":999","tags":["n","m"],"nested":{"score":11,"ok":true},"note":"brace } inside"}
{"id":42,"name":"slash \\ nested score\":123","tags":["p","q"],"nested":{"score":-5,"ok":true},"note":"end"}
"#;

    let serial = ndjson::parse_ndjson_id_score(data);
    assert_eq!(serial.records, 3);
    assert_eq!(serial.sum, 62);

    for threads in [1usize, 2, 8] {
        assert_eq!(
            ndjson::parse_ndjson_id_score_par(data, threads),
            serial,
            "parallel id+score diverged at {threads} threads"
        );
    }
}
