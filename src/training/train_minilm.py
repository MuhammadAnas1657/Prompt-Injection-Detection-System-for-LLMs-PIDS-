"""
Phase 2: Baseline Defense Model Training (MiniLM)
This script trains the MiniLM model on the PIDS dataset for prompt injection detection.
"""
import os
import sys
import time
import json
import warnings
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, 
                          get_linear_schedule_with_warmup)
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

warnings.filterwarnings("ignore")

# Configuration
DATA_DIR = "/kaggle/input/datasets/anasqaiser/abcdefghijkl/PIDS" 
WORKING_DIR = "/kaggle/working/"
MODELS_DIR = os.path.join(WORKING_DIR, "models")
RESULTS_DIR = os.path.join(WORKING_DIR, "results")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

cfg = {
    "model_key": "minilm",
    "hf_name":   "sentence-transformers/all-MiniLM-L6-v2",
    "max_len":   128,
    "batch":     32,
    "grad_acc":  4,
    "lr":        2e-5,
    "epochs":    3,
    "params":    "22M",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

class PromptDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.texts  = df["text"].fillna("").astype(str).tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self): 
        return len(self.texts)
        
    def __getitem__(self, idx):
        encodings = self.tokenizer(
            self.texts[idx], 
            max_length=self.max_len, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def evaluate_model(model, data_loader, device):
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for batch in data_loader:
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())
            
    predictions = [1 if p >= 0.5 else 0 for p in all_probs]
    auc_score = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else None
    
    return {
        "accuracy": round(accuracy_score(all_labels, predictions), 4),
        "f1": round(f1_score(all_labels, predictions, zero_division=0), 4),
        "roc_auc": round(auc_score, 4) if auc_score else None,
    }

def main():
    print("Loading PIDS dataset splits...")
    df_train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df_val   = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    df_test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    tokenizer = AutoTokenizer.from_pretrained(cfg["hf_name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["hf_name"], num_labels=2, ignore_mismatched_sizes=True
    ).to(DEVICE)

    train_ds = PromptDataset(df_train, tokenizer, cfg["max_len"])
    val_ds   = PromptDataset(df_val,   tokenizer, cfg["max_len"])
    test_ds  = PromptDataset(df_test,  tokenizer, cfg["max_len"])

    train_loader = DataLoader(train_ds, batch_size=cfg["batch"], shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch"]*2, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=cfg["batch"]*2, shuffle=False, num_workers=2)

    total_steps = (len(train_loader) // cfg["grad_acc"]) * cfg["epochs"]
    optimizer   = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    scheduler   = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, total_steps//10), num_training_steps=total_steps
    )
    criterion = torch.nn.CrossEntropyLoss()
    scaler    = GradScaler() if DEVICE == "cuda" else None

    best_f1 = 0.0
    best_ckpt = os.path.join(MODELS_DIR, f"{cfg['model_key']}_pids")
    history = []

    print(f"\nTraining Model: {cfg['model_key'].upper()} ({cfg['params']})")
    
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total_loss = 0.0
        start_time = time.time()
        optimizer.zero_grad()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['epochs']}")

        for step, batch in enumerate(progress_bar):
            labels = batch.pop("labels").to(DEVICE)
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            if scaler:
                with autocast():
                    outputs = model(**batch)
                    loss = criterion(outputs.logits, labels) / cfg["grad_acc"]
                scaler.scale(loss).backward()
            else:
                outputs = model(**batch)
                loss = criterion(outputs.logits, labels) / cfg["grad_acc"]
                loss.backward()

            total_loss += loss.item() * cfg["grad_acc"]

            if (step + 1) % cfg["grad_acc"] == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            progress_bar.set_postfix({"Loss": f"{total_loss/(step+1):.4f}"})

        avg_loss = total_loss / len(train_loader)
        print("\nExecuting validation evaluation...")
        val_metrics = evaluate_model(model, val_loader, DEVICE)
        print(f"Epoch {epoch} Results | Loss: {avg_loss:.4f} | Val F1: {val_metrics['f1']:.4f} | Acc: {val_metrics['accuracy']:.4f}")

        history.append({"epoch": epoch, "train_loss": round(avg_loss, 4), **val_metrics})

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            model.save_pretrained(best_ckpt)
            tokenizer.save_pretrained(best_ckpt)
            print(f"[*] New best model saved (F1: {best_f1:.4f}) to {best_ckpt}\n")

    print("\nExecuting final holdout test evaluation...")
    best_model = AutoModelForSequenceClassification.from_pretrained(best_ckpt).to(DEVICE)
    test_metrics = evaluate_model(best_model, test_loader, DEVICE)

    print(f"Final Test Metrics -> F1: {test_metrics['f1']:.4f}, Accuracy: {test_metrics['accuracy']:.4f}, AUC: {test_metrics['roc_auc']:.4f}")

    hist_file = os.path.join(RESULTS_DIR, f"{cfg['model_key']}_results.json")
    with open(hist_file, "w") as f:
        json.dump({
            "model": cfg["model_key"], 
            "hf_name": cfg["hf_name"], 
            "best_val_f1": best_f1, 
            "test_metrics": test_metrics, 
            "history": history, 
            "saved_to": best_ckpt
        }, f, indent=2)

if __name__ == "__main__":
    main()
