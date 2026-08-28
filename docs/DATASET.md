# Dataset Notes

## Source

The notebook uses the **JailBreakV-28K** dataset from Hugging Face:

`JailbreakV-28K/JailBreakV-28k`

The dataset configuration used by the notebook is:

`JailBreakV_28K`

## Expected fields

The notebook relies on fields including:

- `jailbreak_query` — text input
- `policy` — target safety-policy category
- `format` — jailbreak attack format
- `redteam_query` — used in the leakage-focused ablation

The exact schema should be verified by running the dataset-loading cell.

## Local caching

If `jailbreakv28k.parquet` exists in the working directory, the notebook loads it directly. Otherwise it downloads the dataset and creates the local Parquet cache.

The local cache is intentionally not committed to Git.

## Data governance

Before redistributing the dataset or publishing derived records, review the dataset's own license, terms, and responsible-use guidance.
