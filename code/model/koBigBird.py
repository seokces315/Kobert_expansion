import os
import sys

# Get parent folder path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(os.path.dirname(parent_dir))

# Ignore warnings
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

from utils import random_state_init

from data.load import load_data
from torch.utils.data import TensorDataset, DataLoader

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer

from tqdm import tqdm
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

import numpy as np
import torch


class EarlyStopping:
    def __init__(self, patience=3, verbose=True, delta=0.0, path="checkpoint.pt"):
        """
        Args:
            patience (int): 개선되지 않아도 기다리는 에폭 수
            verbose (bool): True면 개선될 때마다 출력
            delta (float): 개선으로 간주할 최소 변화량
            path (str): 모델을 저장할 경로
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss  # 낮을수록 좋은 loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)

        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping Counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """성능이 향상될 때 모델 저장"""
        if self.verbose:
            print(
                f"Validation loss 감소 ({self.val_loss_min:.6f} → {val_loss:.6f}). 모델 저장."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# KoBigBird + Classifier (MLP)
class koBigBirdClassifier(nn.Module):

    # Generator
    def __init__(self, model_id, dropout, hidden_size, num_labels):
        super(koBigBirdClassifier, self).__init__()
        self.backbone = AutoModel.from_pretrained(model_id)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )

    # Forward
    def forward(self, **encoded_input):
        outputs = self.backbone(**encoded_input)
        cls_tokens = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(cls_tokens))
        return logits


# Reproducibility
random_state_init(42)

# GPU settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
korean_dataset = load_data("../../data/KoreanData.csv")
texts = [text for text in korean_dataset["input_text_long"]]
labels = [(label - 1) for label in korean_dataset["difficulty_label"]]

# Load embedding model & tokenizer
model_id = "monologg/kobigbird-bert-base"
model = koBigBirdClassifier(
    model_id=model_id, dropout=0.1, hidden_size=768, num_labels=5
)
model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load optimizer & loss function
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

# Train/Test Split
X_train, X_test, Y_train, Y_test = train_test_split(
    texts, labels, test_size=0.2, stratify=labels, random_state=42
)

# Tokenizing
train_encodings = tokenizer(
    X_train, max_length=2048, padding=True, truncation=True, return_tensors="pt"
)
test_encodings = tokenizer(
    X_test, max_length=2048, padding=True, truncation=True, return_tensors="pt"
)

# Transform into TensorDataset
train_dataset = TensorDataset(
    train_encodings["input_ids"],
    train_encodings["attention_mask"],
    torch.tensor(Y_train),
)
test_dataset = TensorDataset(
    test_encodings["input_ids"], test_encodings["attention_mask"], torch.tensor(Y_test)
)

# DataLoader
batch_size = 4
g = torch.Generator()
g.manual_seed(42)
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, generator=g
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, generator=g
)

# Train Loop
# for epoch in range(3):
#     total_loss = 0
#     correct = 0
#     total = 0

#     for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
#         input_ids, attention_mask, labels = batch
#         input_ids = input_ids.to(device)
#         attention_mask = attention_mask.to(device)
#         labels = labels.to(device)

#         optimizer.zero_grad()
#         logits = model(
#             input_ids=input_ids, attention_mask=attention_mask, labels=labels
#         )
#         loss = criterion(logits, labels)
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()
#         preds = torch.argmax(logits, dim=1)
#         correct += (preds == labels).sum().item()
#         total += labels.size(0)

#         print(
#             f"[Epoch {epoch+1}] Loss: {total_loss/len(train_loader):.4f} | Accuracy: {correct/total:.4f}"
#         )
#         print()

early_stopping = EarlyStopping(patience=2, path="best_model.pt")

for epoch in range(10):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in tqdm(train_loader, desc=f"[Train] Epoch {epoch + 1}"):
        input_ids, attention_mask, labels = [x.to(device) for x in batch]

        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    avg_train_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            val_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    avg_val_loss = val_loss / len(test_loader)
    val_acc = val_correct / val_total

    print(
        f"[Epoch {epoch + 1}] Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f} | "
        f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}"
    )

    early_stopping(avg_val_loss, model)

    if early_stopping.early_stop:
        print("Early stopping triggered.")
        break

# Load best model
model.load_state_dict(torch.load("best_model.pt"))

# Evaluation
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(test_loader):
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        logits = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print("[ 최종 결과 ]")
    print(f"테스트 정확도: {acc:.4f}")
