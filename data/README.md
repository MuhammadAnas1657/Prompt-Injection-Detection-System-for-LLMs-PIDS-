# Dataset Information & Reproduction

Due to GitHub's 100MB file size limit, the raw datasets are not included in this repository. This document explains the origins of our datasets and how you can independently download and reproduce them.

## 1. Primary Training Dataset (PIDS)
The Prompt Injection Detection System (PIDS) dataset is our primary dataset used for fine-tuning the defense models. It consists of highly balanced prompt injection attacks and benign user queries.

- **Source:** Kaggle (Anas Qaiser PIDS Dataset)
- **Size:** ~350,000 samples
- **Structure:** Split into `train.csv`, `val.csv`, and `test.csv` (70/15/15 split).
- **Reproduction:** The dataset must be downloaded from its source and placed in the `data/raw/` directory before running the training scripts.

## 2. Pure Out-Of-Distribution (OOD) Evaluation Dataset
To rigorously evaluate our models against zero-day, never-before-seen threats without any data leakage, we dynamically generate a 100% pure external evaluation dataset.

- **Attacks (Label 1):** 2,500 prompt injection attacks sampled from `ai-ml-ops-eng/tensortrust-datasets` (HuggingFace).
- **Safe Prompts (Label 0):** 2,500 conversational benign prompts sampled from `dair-ai/emotion` (HuggingFace).
- **Reproduction:** You do not need to manually download this data. You can simply run our preprocessing script `data/scripts/build_pure_ood.py`, which will automatically fetch the external data, verify integrity, balance the classes, and generate the `pure_external_ood.csv` file.
