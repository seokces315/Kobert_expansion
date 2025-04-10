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

from utils import set_seed
import torch

from data.load import load_data
from evaluation import manifold_plot

from transformers import AutoModel
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from evaluation import eval_with_classifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from catboost import CatBoostClassifier


# Reproducibility
set_seed(42)

# GPU settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
korean_dataset = load_data("../../data/KoreanData.csv")
texts = [text for text in korean_dataset["input_text_long"]]
labels = [label for label in korean_dataset["difficulty_label"]]

# Load embedding model
model_id = "jinaai/jina-embeddings-v3"
embedding_model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
embedding_model.to(device)

# Create sentence embeddings
print()
print("===========================================================================")
print()
print("< Corpus Vectorization Process >")
print()

embeddings = []
batch_size = 32
for i in tqdm(range(0, len(texts), batch_size)):
    batch_texts = texts[i : i + batch_size]
    embedding = embedding_model.encode(
        batch_texts, task="classification", convert_to_tensor=True, device="cuda"
    )
    # embeddings = embedding_model.encode("Hi", task="classification", max_length=2)
    embeddings.append(embedding.cpu().detach())

# List -> Batch
embeddings = torch.cat(embeddings, dim=0).numpy()
print()
print("[ Processing Done! ]")
print(f"Full batch: {len(embeddings)}")
print(f"Hidden State -> {embeddings[0].shape}")
print()

# Checking generated embeddings
manifold_plot(0, embeddings=embeddings, labels=labels)

# Train/Test Split
X_train, X_test, Y_train, Y_test = train_test_split(
    embeddings,
    labels,
    test_size=0.2,
    stratify=labels,
    random_state=42,
)

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
