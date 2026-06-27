# Dataset Splits

## RetailCabinet-4

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

## SSv2 Resources

`temporal_static_splits/ssv2_splits/train_temporal18.txt` and
`temporal_static_splits/ssv2_splits/validation_temporal18.txt` are the
Something-Something V2 Temporal18 split files reused from the upstream
[TC-CLIP](https://github.com/naver-ai/tc-clip) repository. They are included so
the SSV2-Temporal18 generalization experiments can be reproduced with the same
public split definitions. Raw
[Something-Something V2](https://developer.qualcomm.com/software/ai-datasets/something-something)
videos are not included.
