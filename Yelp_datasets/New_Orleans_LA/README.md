# Model input export

Dynamic event CSVs use four integer columns:

```text
user_id,rel_id,business_id,ts
```

- `user_id` is assigned by first appearance in the full timestamp-sorted
  dynamic event stream and ranges from `0` to `num_users - 1`.
- `business_id` is assigned by first appearance in that same full stream, but
  offset by `num_users`, so the first business has id `num_users`.
- `rel_id` is `review_1star..review_5star => 0..4`, `tip => 5`.
- `ts` is the number of seconds since the first dynamic event in the full city
  dataset, so the first event has `ts = 0`.

`valid_events.csv` and `test_events.csv` are based on the already-clean
transductive splits from `valid.csv` and `test.csv`.

`static_user_friend_edges.csv` has two integer user-id columns and no timestamp.
`business_id_map.csv` includes the Yelp latitude/longitude from business
metadata plus a `geo` column formatted as `latitude,longitude`.
