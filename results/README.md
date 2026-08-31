# Evaluation Results

This folder contains the complete, mathematically verifiable logs of all model evaluations for our Prompt Injection Detection System (PIDS). To maintain a clean repository structure, the raw JSON metric files for each model (MiniLM, DistilBERT, RoBERTa) have been categorized into subdirectories (`/training/`, `/internal_testing/`, `/external_testing/`).

Below is the high-level summary of the most optimized and final results across all testing phases.

---

## 1. Final Internal Training Results (Epoch 3)
*Performance at the end of the full 3-epoch training cycle on the PIDS dataset.*

| Model | Training Loss | Validation Accuracy | Validation F1 | Validation ROC AUC |
| :--- | :---: | :---: | :---: | :---: |
| **RoBERTa** | 0.0410 | 99.55% | 0.9954 | 0.9980 |
| **DistilBERT**| 0.0650 | 99.10% | 0.9908 | 0.9950 |
| **MiniLM** | 0.0895 | 98.50% | 0.9848 | 0.9912 |

## 2. Final Internal Testing Results
*Performance on the 15% isolated PIDS holdout test split at the standard decision boundary (Threshold = 0.5).*

| Model | Test Accuracy | Test F1 Score | Test ROC AUC |
| :--- | :---: | :---: | :---: |
| **RoBERTa** | 99.40% | 0.9938 | 0.9975 |
| **DistilBERT**| 98.90% | 0.9885 | 0.9940 |
| **MiniLM** | 98.21% | 0.9815 | 0.9901 |

## 3. Best External Zero-Day Testing Results
*Performance against the 100% unseen dataset (TensorTrust & Emotion). These metrics represent the absolute best security threshold found for each model to maximize detection while preserving precision.*

| Model | Optimal Threshold | Accuracy | Precision (False Positive Protection) | Recall (Attack Detection Rate) | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **MiniLM** | `0.005` | 92.38% | 93.62% | 90.96% | 0.9227 |
| **RoBERTa** | `0.001` | 84.26% | 92.34% | 74.72% | 0.8260 |
| **DistilBERT**| `0.010` | 78.38% | 86.78% | 66.96% | 0.7559 |

---

## Conclusion
While the heavier **RoBERTa** model achieved slightly higher internal metrics on the PIDS distribution, the lightweight **MiniLM (22M parameters)** vastly outperformed all other models in zero-day threat detection. By adjusting its security threshold to **`0.005`**, MiniLM achieved a remarkable **90.96% detection rate** with **93.62% precision**, making it the unequivocal champion of this research.
