import pandas as pd
from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_ood_dataset(output_path: str, samples_per_class: int = 2500) -> None:
    """
    Constructs a completely unbiased, out-of-distribution (OOD) evaluation dataset 
    by sampling from external HuggingFace repositories.
    
    Args:
        output_path (str): The path to save the generated CSV.
        samples_per_class (int): The number of samples to extract per class to ensure balance.
    """
    logging.info("Initiating OOD dataset construction...")
    
    # 1. Fetch malicious prompt injections (Zero-Day Attacks)
    logging.info(f"Fetching {samples_per_class} adversarial samples from TensorTrust...")
    ds_tt = load_dataset('ai-ml-ops-eng/tensortrust-datasets', split='train')
    df_attacks = pd.DataFrame(ds_tt)[['attacker_input']].rename(columns={'attacker_input': 'text'}).dropna()
    df_attacks['label'] = 1
    df_attacks['source'] = 'tensortrust'
    df_attacks = df_attacks.sample(n=samples_per_class, random_state=42)
    
    # 2. Fetch benign conversational text (Zero-Day Safe Prompts)
    logging.info(f"Fetching {samples_per_class} benign samples from Emotion...")
    ds_em = load_dataset('dair-ai/emotion', split='train')
    df_em = pd.DataFrame(ds_em)[['text']].dropna()
    df_em['label'] = 0
    df_em['source'] = 'emotion'
    df_em = df_em.sample(n=samples_per_class, random_state=42)
    
    # 3. Concatenate and shuffle to ensure random distribution during evaluation
    df_pure = pd.concat([df_attacks, df_em]).sample(frac=1, random_state=42).reset_index(drop=True)
    df_pure.to_csv(output_path, index=False)
    
    logging.info(f"Dataset successfully created at {output_path} with {len(df_pure)} total samples.")

if __name__ == "__main__":
    build_ood_dataset(output_path="pure_external_ood.csv")
