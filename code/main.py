import os
import sys
import warnings

# Get parent folder path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Append to sys.path
sys.path.append(parent_dir)

# Ignore warnings
warnings.filterwarnings("ignore")

from parser import parse_args
from utils import set_seed
import torch

from data.load import load_data
from torch.utils.data import Dataset

from transformers import AutoTokenizer, AutoModel


# Custom dataset for RoBERTa method
class RecursiveChunkDataset(Dataset):
    # Generator
    def __init__(self, tokenized_texts, labels):
        self.tokenized_texts = tokenized_texts
        self.labels = labels

    # Length getter

    # Getter


# Main flow
def main(args):

    # Reproducibility
    set_seed(args.seed)

    # GPU settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    dataset_id = f"../data/{args.dataset}"
    csat_kor_dataset = load_data(dataset_id)

    # Data pre-processing
    if args.model_type == "roberta":
        # Load tokenizer
        model_id = "klue/roberta-large"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id)
        model.to(device)

        # Tokenize function
        def recursive_tokenize(text):
            encoded_text = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=512,
                stride=128,
                return_tensors="pt",
                return_overflowing_tokens=True,
            )
            return encoded_text

        # Splitting, Tokenizing text
        csat_kor_dataset["text"] = csat_kor_dataset["text"].map(recursive_tokenize)
        print(csat_kor_dataset["text"][0])

        # Generate chunk-wise embeddings
        print()
        print("====================================================================")
        print()
        print("< Chunk-wise Embedding >")
        print()

        brk = 0
        dataset_length = len(csat_kor_dataset["text"])
        model.eval()
        for text_dict in csat_kor_dataset["text"]:
            # Logging
            if brk % 100 == 0:
                print(f"[Current/Total Corpus] : {brk}/{dataset_length}")
                print()

            # Model forwarding
            text_dict = {k: v.to(device) for k, v in text_dict.items()}
            with torch.no_grad():
                output = model(**text_dict)

            brk += 1
        print(f"[Current/Total Corpus] : {brk}/{dataset_length}")
        print()

    # else:


if __name__ == "__main__":
    args = parse_args()
    main(args)
