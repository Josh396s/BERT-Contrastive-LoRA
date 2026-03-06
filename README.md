# PA4 - Transformers for Amazon Scenario Classification

This project explores the use of transformer-based models, specifically BERT, for the task of scenario classification. The main focus is on fine-tuning a pre-trained BERT model, along with experiments that incorporate custom fine-tuning strategies and contrastive learning techniques. We compare the performance of a baseline model, custom fine-tuning approaches, and contrastive learning methods like SimCLR and SupContrast.

## Models and Techniques:
- **Baseline Model**: Fine-tune a pre-trained BERT model for text classification using cross-entropy loss.
- **Custom Fine-tuning**: Implement advanced techniques like Layer-wise Learning Rate Decay (LLRD) and Warm-up Scheduler to improve model performance.
- **Contrastive Learning**: Apply contrastive learning approaches such as SupContrast and SimCLR to enhance feature learning through supervised and unsupervised contrastive loss functions.

## Running the Project

To run the project with a specific configuration, use the following command format:
python main.py --task <model_name> [additional_arguments]

Examples: 
- `python main.py --task baseline --embed-dim 768 --learning-rate 1e-4 --hidden-dim 768 --drop-rate 0.7`
- `python main.py --task baseline --embed-dim 768 --learning-rate 1e-4 --hidden-dim 768  --drop-rate 0.7 --lora True --lora-rank 8 --lora-targets query value --lora-dropout 0.1`
- `python main.py --task custom --embed-dim 768 --learning-rate 1e-4 --hidden-dim 768 --drop-rate 0.7`
- `python main.py --task supcon --embed-dim 768 --learning-rate 1e-4 --hidden-dim 768 --drop-rate 0.7`


Hyperparameters:
You can adjust the hyperparameters needed, such as:
- --embed-dim: Embedding dimension (e.g., 768 for BERT).
- --learning-rate: Learning rate for optimization (e.g., 1e-4).
- --hidden-dim: Hidden layer dimension (e.g., 768).
- --drop-rate: Dropout rate for regularization (e.g., 0.7).
- --lora: Set to True to use LoRA fine-tuning.
- --lora-rank: Rank value for LoRA (e.g., 8).
- --lora-targets: Layers to apply LoRA (e.g., query value).
- --lora-dropout: Dropout rate for LoRA layers.

