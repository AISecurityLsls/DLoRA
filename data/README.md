# Data

The CL_Benchmark Long-Sequence datasets are not committed to this repository.
Download the four tasks used by the default setup and place each task under:

```text
data/Long_Sequence/<task>/
```

Each task directory contains `train.json`, `dev.json`, `test.json`, and
`labels.json`. The training code reads these files directly. To add another
CL_Benchmark task, copy its directory under `Long_Sequence/` and, if needed,
add the corresponding path override to `configs/paths.local.json`.

Tasks used by the default setup: `agnews`, `imdb`, `mnli`, and `multirc`.
