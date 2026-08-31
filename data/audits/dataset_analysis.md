# Dataset Audit Report

This document provides mathematical and statistical verification of the integrity of the datasets used in our Prompt Injection Detection System (PIDS) research.

## 1. Primary Dataset (PIDS) Integrity Audit
Before training any models, the internal dataset was analyzed for class imbalance, data corruption (missing rows), and maximum token length.

### Structural Integrity
- **Total Rows Evaluated**: 350,000
- **Missing Values (NaNs)**: 0
- **Duplicate Rows**: Removed during preprocessing.

### Class Distribution (Balance)
The training and validation sets were meticulously balanced to ensure no algorithmic bias towards the majority class.

| Class | Label | Count | Percentage |
| :--- | :---: | :---: | :---: |
| **Safe (Benign)** | `0` | 175,000 | 50.0% |
| **Prompt Injection** | `1` | 175,000 | 50.0% |

### Tokenization Analysis
To determine the optimal context window (`max_length`) for our small language models (SLMs), we analyzed the token length distribution of the dataset using the LLaMA tokenizer:
- **99th Percentile Token Length**: ~240 tokens
- **Max Length Selected**: 256 tokens

*Conclusion: Truncating prompts at 256 tokens captures >99% of the semantic meaning of the dataset without unnecessarily increasing computational overhead during inference.*

## 2. External Out-Of-Distribution (OOD) Dataset Audit
To prevent the models from simply memorizing the PIDS dataset, the final multi-threshold evaluations are conducted on a completely distinct, dynamically generated dataset.

### Structural Integrity
- **Source Repositories**: `ai-ml-ops-eng/tensortrust-datasets` (Attacks) and `dair-ai/emotion` (Benign)
- **Total Rows Evaluated**: 5,000 (Randomly sampled to fit constraints)

### Class Distribution (Balance)
| Class | Label | Count | Percentage |
| :--- | :---: | :---: | :---: |
| **Safe (Emotion Dataset)** | `0` | 2,500 | 50.0% |
| **Prompt Injection (TensorTrust)** | `1` | 2,500 | 50.0% |

*Conclusion: The 50/50 balance ensures that Precision and Recall metrics are authentic and not artificially inflated by a majority class.*
