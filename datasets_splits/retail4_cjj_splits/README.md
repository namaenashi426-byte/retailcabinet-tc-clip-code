# RetailCabinet-4 Splits

These files define the fixed class-stratified video-level split used in the paper:

- `train.txt`: 1200 samples.
- `val.txt`: 301 samples.
- `test.txt`: 368 samples.

Each line has the format:

```text
relative/video/path.mp4 class_id
```

Class IDs are defined in `labels/retail4_cjj_labels.csv`.

The raw videos are not included because they contain human-participant recordings.
