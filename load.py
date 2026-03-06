from transformers import BertTokenizer
from datasets import load_dataset

def load_data():
    dataset = load_dataset("mteb/amazon_massive_scenario", "en")
    print(dataset)
    return dataset

def load_tokenizer(args):
    # Load bert tokenizer from pretrained "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    return tokenizer