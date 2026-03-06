import argparse
import os
import json

def load_config(config_path):
    """Load experiment configurations from a JSON file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = json.load(f)

    # Convert dictionary to argparse.Namespace
    return argparse.Namespace(**config_dict)

def params():
    parser = argparse.ArgumentParser()

    # Experiment options
    parser.add_argument("--task", default="baseline", type=str,\
                help="baseline is fine-tuning bert for classification;\n\
                      tune is advanced techiques to fine-tune bert;\n\
                      constast is contrastive learning method")
                      #choices=['baseline','tune','supcon'])

    # optional fine-tuning techiques parameters
    parser.add_argument("--reinit_n_layers", default=0, type=int, 
                help="number of layers that are reinitialized. Count from last to first.")
    
    # Others
    parser.add_argument("--input-dir", default='assets', type=str, 
                help="The input training data file (a text file).")
    parser.add_argument("--output-dir", default='results', type=str,
                help="Output directory where the model predictions and checkpoints are written.")
    parser.add_argument("--model", default='bert', type=str,
                help="The model architecture to be trained or fine-tuned.")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--dataset", default="amazon", type=str,
                help="dataset", choices=['amazon'])
    

    # Key settings
    parser.add_argument("--ignore-cache", action="store_true",
                help="Whether to ignore cache and create a new input data")
    parser.add_argument("--debug", action="store_true",
                help="Whether to run in debug mode which is exponentially faster")
    parser.add_argument("--do-train", action="store_true",
                help="Whether to run training.")
    parser.add_argument("--do-eval", action="store_true",
                help="Whether to run eval on the dev set.")
    
    # Hyper-parameters for tuning
    parser.add_argument("--batch-size", default=16, type=int,
                help="Batch size per GPU/CPU for training and evaluation.")
    parser.add_argument("--learning-rate", default=1e-4, type=float,
                help="Model learning rate starting point.")
    parser.add_argument("--hidden-dim", default=10, type=int,
                help="Model hidden dimension.")
    parser.add_argument("--drop-rate", default=0.9, type=float,
                help="Dropout rate for model training")
    parser.add_argument("--embed-dim", default=10, type=int,
                help="The embedding dimension of pretrained LM.")
    parser.add_argument("--adam-epsilon", default=1e-8, type=float,
                help="Epsilon for Adam optimizer.")
    parser.add_argument("--n-epochs", default=10, type=int,
                help="Total number of training epochs to perform.")
    parser.add_argument("--max-len", default=20, type=int,
                help="maximum sequence length to look back")
    parser.add_argument("--weight-decay", default=0.01, type=float,
                help="weight decay for LLRD optimizer")

    # LoRA Settings:
    parser.add_argument("--lora", default=False, type=bool,
                help="Flag for using LoRA")
    parser.add_argument("--lora-rank", default=16, type=int,
                help="Rank for to use for LoRA")
    parser.add_argument(
    "--lora-targets", nargs="+", default=["query", "value"],
    help="Modules that LoRA will be applied to"
)
    parser.add_argument("--lora-dropout", default=0.1, type=float,
                help="Dropout for LoRA layers")
    

    ## supcon
    parser.add_argument("--loss_type", default='supcon', type=str,
                help="simclr or supcon")


    args = parser.parse_args()
    return args



