# BERT Fine-Tuning with Supervised Contrastive Learning and LoRA

This repository contains a research-focused implementation for enhancing intent and scenario classification. By combining **Supervised Contrastive Learning (SupCon)**, **LoRA (Low-Rank Adaptation)**, and **Layer-wise Learning Rate Decay (LLRD)**, this pipeline achieves a robust representation of textual data.

## Performance Highlights
* **Metric Improvement:** Achieved a **1.2% F1-score lift** on the Amazon Massive dataset compared to standard BERT fine-tuning baselines.
* **Efficiency:** Utilized LoRA to significantly reduce trainable parameters while maintaining high-performance inference capabilities.
* **Robustness:** Integrated **SupCon** and **SimCLR** logic to improve cluster separation in the embedding space for edge-case intent scenarios.

## Key Features
- **Advanced Fine-Tuning:** Implements Layer-wise Learning Rate Decay (LLRD) to prevent catastrophic forgetting in early transformer layers.
- **Contrastive Learning:** Includes custom `loss.py` implementing Supervised Contrastive Loss for optimized feature representation.
- **Parameter Efficiency:** Full PEFT integration via LoRA.
- **Production Ready:** Clean, modular Python structure with full support for configuration via `config.json`.

## Repository Structure
* `model.py`: Core architecture combining BERT with contrastive projection heads and LoRA adapters.
* `loss.py`: Implementation of SupCon and SimCLR loss functions.
* `main.py`: Training and evaluation orchestration.
* `dataloader.py`: Specialized data pipeline for the Amazon Massive dataset.

## Installation & Usage

1. **Clone the repo:**
   ```bash
   git clone [https://github.com/Josh396s/BERT-Contrastive-LoRA.git](https://github.com/Josh396s/BERT-Contrastive-LoRA.git)
   cd BERT-Contrastive-LoRA
   ```
2. **Setup Environment**
  ```bash
  pip install -r requirements.txt
  ```
3. **Run Training:**
  ```bash
  python main.py --config config.json
  ```
## Dataset
This project primarily utilizes the Amazon Massive dataset, a multilingual mid-range intentional dataset for NLU.
```bash
  Dataset: https://huggingface.co/datasets/AmazonScience/massive
```
