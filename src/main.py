import os
import sys
import warnings
import logging

# Get parent folder path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Append to sys.path
sys.path.append(parent_dir)

# Ignore warnings
warnings.filterwarnings("ignore")
# Set log level lower
logging.getLogger().setLevel(logging.ERROR)


from parser import parse_args
from utils import set_seed

import torch
import torch.nn as nn

from data.load import load_data
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformers import logging, AutoTokenizer, AutoModel
from model.jinaEmb import JinaEmbeddingClassifier
from model.koBigBird import koBigBirdClassifier
from train import train_model, eval_model, get_embeddings, eval_with_classifier

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from catboost import CatBoostClassifier

# Set log level lower
logging.set_verbosity_error()


# Tokenized custom dataset
class PreTokenizedDataset(Dataset):
    # Generator
    def __init__(self, input_dicts, labels):
        self.input_dicts = input_dicts
        self.labels = labels

    # Length getter
    def __len__(self):
        return len(self.labels)

    # Getter
    def __getitem__(self, idx):
        input_dict = self.input_dicts[idx]
        label = self.labels[idx]

        return {
            "input_ids": input_dict["input_ids"],
            "attention_mask": input_dict["attention_mask"],
            "labels": torch.tensor(label, dtype=torch.long),
        }


# Function for dynamic padding
def collate_fn(batch):
    # Batch sample -> Key list
    input_ids = [item["input_ids"].squeeze(0) for item in batch]
    attention_mask = [item["attention_mask"].squeeze(0) for item in batch]
    labels = [item["labels"] for item in batch]

    # Padding
    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=1)
    padded_attention_mask = pad_sequence(
        attention_mask, batch_first=True, padding_value=0
    )

    # Stacking
    labels = torch.stack(labels)

    return {
        "input_ids": padded_input_ids,
        "attention_mask": padded_attention_mask,
        "labels": labels,
    }


# Function to split dataset
def split_dataset(inputs, labels, seed, val_size=0.2, test_size=0.2):
    # Train/Test split
    train_inputs, vt_inputs, train_labels, vt_labels = train_test_split(
        inputs,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )

    # Valid/Test split
    val_inputs, test_inputs, val_labels, test_labels = train_test_split(
        vt_inputs,
        vt_labels,
        test_size=test_size / (val_size + test_size),
        random_state=seed,
        stratify=vt_labels,
    )

    # Reset index & Return
    ret_splits = [
        train_inputs,
        train_labels,
        vt_inputs,
        vt_labels,
        val_inputs,
        val_labels,
        test_inputs,
        test_labels,
    ]
    ret_splits = [ret.reset_index(drop=True) for ret in ret_splits]

    return ret_splits


# Main flow
def main(args):

    # Reproducibility
    set_seed(args.seed)

    # GPU settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    dataset_id = f"../data/{args.dataset}"
    csat_kor_dataset = load_data(dataset_id)

    # Branching by model type
    if args.model_type == "roberta":
        # Load tokenizer & model
        model_id = "klue/roberta-large"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id)
        model.to(device)

    elif args.model_type == "bigbird":
        # Load tokenizer & model
        model_id = "monologg/kobigbird-bert-base"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        se_model = AutoModel.from_pretrained(model_id, trust_remote_code=True)

        # Tokenize function
        def tokenize(text):
            encoded_text = tokenizer(
                text,
                padding=False,
                truncation=True,
                max_length=2048,
                return_tensors="pt",
            )
            return encoded_text

        # Splitting, Tokenizing text
        csat_kor_dataset["text"] = csat_kor_dataset["text"].map(tokenize)

        # Load model-specific dataset
        train_inputs, train_labels, val_inputs, val_labels, test_inputs, test_labels = (
            split_dataset(
                csat_kor_dataset["text"],
                csat_kor_dataset["label"],
                seed=args.seed,
                val_size=0.2,
                test_size=0.2,
            )
        )
        
        train_dataset = PreTokenizedDataset(train_inputs, train_labels)
        val_dataset = PreTokenizedDataset(val_inputs, val_labels)
        test_dataset = PreTokenizedDataset(test_inputs, test_labels)

        # Prepare dataloader
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        valid_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        # [ Training Process ]
        print()
        print("[ Training Process ]")
        print()
        model = koBigBirdClassifier(
            se_model, dropout=0.1, hidden_size=768, num_classes=5
        )
        model = model.to(device)
        trained_model = train_model(
            model, args.epoch, train_loader=train_loader, valid_loader=test_loader
        )

        print()
        print("< Test Data Evaluation >")
        print()
        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc, test_f1, test_auroc = eval_model(
            trained_model, dataloader=test_loader, criterion=criterion
        )
        print(
            f"Test - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}, F1 Score: {test_f1:.4f}, AUROC: {test_auroc:.4f}"
        )

        # [ Additional Process ]
        print()
        print("< + 5 ML Classifiers >")
        print()
        ft_embedding_model = trained_model.embedding_model.to(device)
        ft_embedding_model.eval()

        X_train, y_train = get_embeddings(ft_embedding_model, dataloader=train_loader)
        X_test, y_test = get_embeddings(ft_embedding_model, dataloader=test_loader)
        train_set = (X_train, y_train)
        test_set = (X_test, y_test)

        model_dict = {
            "LightGBM": LGBMClassifier(random_state=42, verbose=0),
            "RandomForest": RandomForestClassifier(random_state=42, verbose=0),
            "LogisticRegression": LogisticRegression(random_state=42, verbose=0),
            "SVC": SVC(probability=True, random_state=42, verbose=0),
            "CatBoost": CatBoostClassifier(random_state=42, verbose=0),
        }
        for model_id, model_instance in model_dict.items():
            eval_with_classifier(model_id, model_instance, train_set, test_set)

    else:
        # Load tokenizer & sentence embedding model
        model_id = "jinaai/jina-embeddings-v3"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        se_model = AutoModel.from_pretrained(model_id, trust_remote_code=True)

        # Tokenize function
        def tokenize(text):
            encoded_text = tokenizer(
                text,
                padding=False,
                truncation=True,
                max_length=3072,
                return_tensors="pt",
            )
            return encoded_text

        # Splitting, Tokenizing text
        csat_kor_dataset["text"] = csat_kor_dataset["text"].map(tokenize)

        # Load model-specific dataset
        train_inputs, train_labels, val_inputs, val_labels, test_inputs, test_labels = (
            split_dataset(
                csat_kor_dataset["text"],
                csat_kor_dataset["label"],
                seed=args.seed,
                val_size=0.2,
                test_size=0.2,
            )
        )
        train_dataset = PreTokenizedDataset(train_inputs, train_labels)
        val_dataset = PreTokenizedDataset(val_inputs, val_labels)
        test_dataset = PreTokenizedDataset(test_inputs, test_labels)

        # Prepare dataloader
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        valid_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        # [ Training Process ]
        print()
        print("[ Training Process ]")
        print()
        model = JinaEmbeddingClassifier(
            se_model, dropout=0.1, hidden_size=1024, num_classes=5
        )
        model = model.to(device)
        trained_model = train_model(
            model, args.epoch, train_loader=train_loader, valid_loader=valid_loader
        )

        print()
        print("< Test Data Evaluation >")
        print()
        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc, test_f1, test_auroc = eval_model(
            trained_model, dataloader=test_loader, criterion=criterion
        )
        print(
            f"Test - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}, F1 Score: {test_f1:.4f}, AUROC: {test_auroc:.4f}"
        )

        # [ Additional Process ]
        print()
        print("< + 5 ML Classifiers >")
        print()
        ft_embedding_model = trained_model.embedding_model.to(device)
        ft_embedding_model.eval()

        X_train, y_train = get_embeddings(ft_embedding_model, dataloader=train_loader)
        X_test, y_test = get_embeddings(ft_embedding_model, dataloader=test_loader)
        train_set = (X_train, y_train)
        test_set = (X_test, y_test)

        model_dict = {
            "LightGBM": LGBMClassifier(random_state=42, verbose=0),
            "RandomForest": RandomForestClassifier(random_state=42, verbose=0),
            "LogisticRegression": LogisticRegression(random_state=42, verbose=0),
            "SVC": SVC(probability=True, random_state=42, verbose=0),
            "CatBoost": CatBoostClassifier(random_state=42, verbose=0),
        }
        for model_id, model_instance in model_dict.items():
            eval_with_classifier(model_id, model_instance, train_set, test_set)


if __name__ == "__main__":
    args = parse_args()
    main(args)
