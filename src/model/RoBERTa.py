import os
import sys

# Get parent folder path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(os.path.dirname(parent_dir))

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")

from utils import set_seed
import torch
import torch.nn as nn

from data.load import load_data
from torch.utils.data import TensorDataset, DataLoader

from nltk.tokenize import sent_tokenize
import nltk

from transformers import BertTokenizer, BertModel
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

from train import eval_with_classifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from catboost import CatBoostClassifier


# Custom Encoder Class
class TransformerKoreanEncoder(nn.Module):

    # Generator
    def __init__(self, input_dim, hidden_dim, num_heads, dropout, num_layers):
        super().__init__()
        encoder_layers = TransformerEncoderLayer(
            d_model=input_dim,
            dim_feedforward=hidden_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = TransformerEncoder(encoder_layers, num_layers)

    # Forward
    def forward(self, embedding_tensors, padding_mask=None):
        encoded_output = self.transformer(
            embedding_tensors, src_key_padding_mask=padding_mask
        )

        # Mean pooling with padding mask
        if padding_mask is not None:
            broadcast_mask = ~padding_mask.unsqueeze(-1)
            masked_embeddings = encoded_output * broadcast_mask
            corpus_embeddings = masked_embeddings.sum(dim=1) / broadcast_mask.sum(dim=1)
        else:
            corpus_embeddings = encoded_output.mean(dim=1)

        return corpus_embeddings


nltk.download("punkt_tab")

# Reproducibility
set_seed(42)

# GPU settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
korean_dataset = load_data("../../data/KoreanData.csv")

# Sentence splitting
korean_dataset["text"] = korean_dataset["text"].apply(sent_tokenize)

# Load embedding tokenizer/model
tokenizer = BertTokenizer.from_pretrained("monologg/kobert")
model = BertModel.from_pretrained("monologg/kobert")
model.to(device)

# Generate sentence-wise embeddings
corpus_embeddings = []
brk = 0
dataset_length = len(korean_dataset["text"])
print()
print("===========================================================================")
print()
print("< Corpus Vectorization Process >")
print()
for corpus in korean_dataset["text"]:
    if brk % 100 == 0:
        print(f"[Current Corpus/Total Corpus] => {brk} / {dataset_length}")
        print()
    # Tokenizing
    input_ids = tokenizer(
        corpus,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    input_ids = {k: v.to(device) for k, v in input_ids.items()}

    # Model forwarding
    with torch.no_grad():
        output = model(**input_ids)

    # CLS token extracting
    cls_tokens = output.last_hidden_state[:, 0, :]
    sentence_embeddings = cls_tokens.cpu().detach()
    corpus_embeddings.append(sentence_embeddings)

    brk += 1

print(f"[Current Corpus/Total Corpus] => {dataset_length} / {dataset_length}")
print()

# List -> Batch
padded_embeddings = pad_sequence(corpus_embeddings, batch_first=True)
full_batch, seq_length, hidden_size = padded_embeddings.shape

# Generate positional encoding
position_ids = torch.arange(seq_length).unsqueeze(0)
positional_embeddings = nn.Embedding(seq_length, hidden_size)(position_ids)
positional_embeddings = positional_embeddings.expand(
    full_batch, seq_length, hidden_size
)
padded_embeddings = padded_embeddings + positional_embeddings

# Generate padding mask
# Padding: True, Actual: False
padding_mask = torch.ones(padded_embeddings.shape[:2], dtype=torch.bool)
for idx, emb in enumerate(corpus_embeddings):
    padding_mask[idx, : emb.shape[0]] = False

# Train/Test Split
train_embs, test_embs, train_masks, test_masks, train_labels, test_labels = (
    train_test_split(
        padded_embeddings,
        padding_mask,
        korean_dataset["label"].to_numpy(),
        test_size=0.2,
        stratify=korean_dataset["label"],
        random_state=42,
    )
)

# Transform into TensorDataset
train_dataset = TensorDataset(train_embs, train_masks, torch.tensor(train_labels))
test_dataset = TensorDataset(test_embs, test_masks, torch.tensor(test_labels))

# DataLoader
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# TransformerKoreanEncoder
kor_encoder = TransformerKoreanEncoder(
    input_dim=768,
    hidden_dim=768,
    num_heads=12,
    dropout=0.1,
    num_layers=4,
)
kor_encoder.to(device)

print("===========================================================================")
print()
print("< Hierarchical BERT - Transformer >")
print()

# Generate train embeddings & labels by batch size
all_input_train_embeddings = []
all_input_train_labels = []
for train_batch in train_loader:
    train_batch = [t.to(device) for t in train_batch]
    train_batch_embs, train_batch_masks, train_batch_labels = train_batch
    train_input_embeddings = kor_encoder(train_batch_embs, train_batch_masks)
    all_input_train_embeddings.append(train_input_embeddings.cpu().detach())
    all_input_train_labels.append(train_batch_labels.cpu().detach())

# Change into full-batch numpy ndarry
all_input_train_embeddings = torch.cat(all_input_train_embeddings, dim=0)
all_input_train_labels = torch.cat(all_input_train_labels, dim=0)
X_train = all_input_train_embeddings.numpy()
Y_train = all_input_train_labels.numpy()

# Generate test embeddings & labels by batch size
all_input_test_embeddings = []
all_input_test_labels = []
for test_batch in test_loader:
    test_batch = [t.to(device) for t in test_batch]
    test_batch_embs, test_batch_masks, test_batch_labels = test_batch
    test_input_embeddings = kor_encoder(test_batch_embs, test_batch_masks)
    all_input_test_embeddings.append(test_input_embeddings.cpu().detach())
    all_input_test_labels.append(test_batch_labels.cpu().detach())

# Change into full-batch numpy ndarry
all_input_test_embeddings = torch.cat(all_input_test_embeddings, dim=0)
all_input_test_labels = torch.cat(all_input_test_labels, dim=0)
X_test = all_input_test_embeddings.numpy()
Y_test = all_input_test_labels.numpy()

# Define classifier models
model_dict = {
    "LightGBM": LGBMClassifier(random_state=42, verbose=0),
    "RandomForest": RandomForestClassifier(random_state=42, verbose=0),
    "LogisticRegression": LogisticRegression(random_state=42, verbose=0),
    "SVC": SVC(probability=True, random_state=42, verbose=0),
    "CatBoost": CatBoostClassifier(random_state=42, verbose=0),
}

# Train & Evaluation
print("===========================================================================")
print()
print("< Result >")
train_set = (X_train, Y_train)
test_set = (X_test, Y_test)
for model_id, model_instance in model_dict.items():
    eval_with_classifier(model_id, model_instance, train_set, test_set)
