# Prompt Injection Detection System for LLMs (PIDS)

## 1. Overview and Objective
This repository contains the full methodology, datasets, training scripts, and evaluation metrics for our Prompt Injection Detection System (PIDS). 

The primary objective of this research is to secure base Large Language Models (LLMs), such as LLaMA, against sophisticated prompt injection attacks in complex, simulated environments like the **AgentDojo** framework. We evaluate the vulnerability of the base model first without any defense, and then protect it using fine-tuned, small language models (MiniLM, DistilBERT, and RoBERTa).

## 2. Evaluation Methodology

### Phase 1: Baseline Evaluation (No Defense)
- **Target**: Evaluate the base LLM in the AgentDojo environment without any security filters.
- **Purpose**: Establish a baseline of vulnerability to prompt injections, specifically focusing on unauthorized actions like data exfiltration, email forwarding, and instruction overrides.

### Phase 2: Defended Evaluation
- **Target**: Evaluate the LLM protected by our fine-tuned defense models.
- **Implementation**: The defense model is implemented at two critical checkpoints (Gates):
  - **Gate 1 (User Security)**: Scans the initial user prompt/input for malicious intent or injection patterns before it reaches the main LLM agent.
  - **Gate 2 (Tool/Data Fetch Security)**: Scans the output of tools (e.g., fetching emails, reading files, executing workspace commands) for indirect prompt injections before that data is fed back into the LLM's context window.

## 3. Data Splitting & Task Execution Strategy
To ensure we can effectively retrain and rigorously test our models without data leakage, we enforce a strict **70/30 Task Split** across all AgentDojo workspaces (banking, travel, workspace, etc.):

- **70% Execution (Evaluation Set)**: Only 70% of the available tasks are executed during the initial evaluation phases. This data is used to compile successes and failures.
- **30% Holdout (Test Set)**: The remaining 30% of tasks are strictly isolated. They are not executed during the initial evaluation and are reserved purely for testing the final, retrained models.

## 4. Error Logging & Analysis
During the 70% Evaluation Set execution, the system tracks the success and failure rates of the agent meticulously:

1. **Failure Compilation (`failed_evaluations.csv`)**
   - Captures any scenario where the model produces incorrect results, falls victim to a prompt injection, or blocks a benign prompt (false positive).
   - Used specifically to analyze weaknesses and fine-tune the defense models.
2. **Success Compilation (`successful_evaluations.csv`)**
   - Captures any scenario where the model correctly executes a benign task or successfully blocks an injection without disrupting the normal conversational flow.

## 5. Retraining and Final Validation
1. **Retraining**: Data compiled from the failure logs is used to re-evaluate our approach and fine-tune the defense models to patch discovered vulnerabilities.
2. **Final Validation (The 30% Set)**: Once retrained, we execute the **30% Holdout Test Set**. This proves the retrained model has genuinely generalized to zero-day threats and has not simply memorized the failure cases from the evaluation set.
