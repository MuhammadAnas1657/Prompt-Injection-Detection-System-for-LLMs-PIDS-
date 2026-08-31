"""
Phase 3: Out-of-Distribution (OOD) Evaluation (DistilBERT)
Evaluates the fine-tuned DistilBERT model on a dynamically generated, 
100% unseen dataset to test zero-day prompt injection detection capabilities.
"""
import os
import sys
import json
import warnings
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

warnings.filterwarnings("ignore")

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OOD_DATA_FILE = os.path.join(BASE_DIR, "data", "pure_external_ood.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models", "distilbert_pids")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
MAX_LEN = 256

def main():
    print(f"\n[{'='*60}]")
    print(" ZERO-DAY THREAT EVALUATION: DISTILBERT ")
    print(f"[{'='*60}]\n")

    if not os.path.exists(OOD_DATA_FILE):
        print(f"[ERROR] OOD dataset not found at {OOD_DATA_FILE}")
        sys.exit(1)

    df_ood = pd.read_csv(OOD_DATA_FILE)
    texts = df_ood["text"].fillna("").astype(str).tolist()
    labels = df_ood["label"].tolist()

    print(f"Loading tokenizer and model from {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
    model.eval()

    all_probs = []
    
    print("\nExecuting inference on unseen data...")
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Evaluating"):
            batch_texts = texts[i:i+BATCH_SIZE]
            inputs = tokenizer(
                batch_texts, return_tensors="pt", truncation=True, 
                max_length=MAX_LEN, padding=True
            ).to(DEVICE)
            
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())

    print("\nGenerating Multi-Threshold Security Analysis...")
    auc = roc_auc_score(labels, all_probs) if len(set(labels)) > 1 else None
    
    # Extreme Threshold Analysis for conservative security gating
    thresholds = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    results = {
        "model": "DistilBERT",
        "dataset": "pure_external_ood",
        "roc_auc": round(auc, 4) if auc else None,
        "thresholds": {}
    }

    for t in thresholds:
        preds = (np.array(all_probs) >= t).astype(int)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, zero_division=0)
        prec = precision_score(labels, preds, zero_division=0)
        rec = recall_score(labels, preds, zero_division=0)
        
        results["thresholds"][str(t)] = {
            "accuracy": round(acc, 4),
            "f1": round(f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4)
        }
        print(f"Threshold: {t:<5} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")

    results_file = os.path.join(RESULTS_DIR, "distilbert_ood_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"\n[SUCCESS] Multi-threshold results saved to {results_file}")

if __name__ == "__main__":
    main()
