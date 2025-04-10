# %% 시드 고정
import os
import numpy as np
import random
import torch


def set_seed(seed=42):
    # os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed(42)

# %% 데이터 로드
import pandas as pd

df = pd.read_csv("KoreanData.csv")
df = df[df["input_text_long"].notnull()]
df = df[df["answer_rate"].notnull()]
df["answer_rate"] = df["answer_rate"].astype(float)

# %% 텍스트 전처리
import re


def preprocess(text):
    text = re.sub(r"[^\w\s]", "", text)
    return text


texts = [preprocess(t) for t in df["input_text_long"].tolist()]

# %% TF-IDF 벡터화
from sklearn.feature_extraction.text import TfidfVectorizer

print("🔍 TF-IDF 벡터화 중...")
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
embeddings = tfidf_vectorizer.fit_transform(texts).toarray()
print(f"✅ 임베딩 shape: {embeddings.shape}")


# %% 등급 라벨 생성
def answer_rate_to_class(rate):
    if rate >= 90:
        return 5  # 하
    elif rate >= 80:
        return 4  # 중하
    elif rate >= 60:
        return 3  # 중
    elif rate >= 50:
        return 2  # 중상
    else:
        return 1  # 상


df["label"] = df["answer_rate"].apply(answer_rate_to_class)


# %% Feature Space 시각화
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def manifold_plot(method, embeddings, labels):

    # PCA into 2D space
    if method == 0:
        pca = PCA(n_components=2, random_state=42)
        reduced_embeddings = pca.fit_transform(embeddings)
    elif method == 1:
        tSNE = TSNE(n_components=2, random_state=42)
        reduced_embeddings = tSNE.fit_transform(embeddings)

    # Visualization
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=reduced_embeddings[:, 0],
        y=reduced_embeddings[:, 1],
        hue=labels,
        palette="tab10",
        s=100,
        alpha=0.7,
    )
    plt.title("Visualization of Embeddings with Difficulty Labels")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(title="Difficulty Label")

    if method == 0:
        plt.savefig("PCA.png")
    elif method == 1:
        plt.savefig("tSNE.png")


# Checking generated embeddings
manifold_plot(0, embeddings=embeddings, labels=df["label"].values)

# %% 데이터 분할
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    embeddings, df["label"].values, test_size=0.2, random_state=42, stratify=df["label"]
)

# %% 모델 리스트
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from catboost import CatBoostClassifier

model_dict = {
    "LightGBM": LGBMClassifier(random_state=42, verbose=0),
    "RandomForest": RandomForestClassifier(random_state=42, verbose=0),
    "LogisticRegression": LogisticRegression(random_state=42, verbose=0),
    "SVC": SVC(probability=True, random_state=42, verbose=0),
    "CatBoost": CatBoostClassifier(random_state=42, verbose=0),
}

# %% 평가 함수
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score


def eval_with_classifier(model_id, model_instance, train_set, test_set):

    # Model fitting
    X_train, Y_train = train_set
    model_instance.fit(X_train, Y_train)

    # Evaluate with test data
    X_test, Y_test = test_set
    test_predicted_labels = model_instance.predict(X_test)

    # Applying given metrics
    # 1. Accuracy
    accuracy = accuracy_score(Y_test, test_predicted_labels)

    # 2. F1 Score ("macro", "weighted)
    f1_macro = f1_score(Y_test, test_predicted_labels, average="macro")
    f1_weighted = f1_score(Y_test, test_predicted_labels, average="weighted")

    # 3. AUROC
    test_predicted_proba = model_instance.predict_proba(X_test)
    try:
        auc_macro = roc_auc_score(
            Y_test, test_predicted_proba, multi_class="ovr", average="macro"
        )
        auc_weighted = roc_auc_score(
            Y_test, test_predicted_proba, multi_class="ovr", average="weighted"
        )
    except ValueError:
        auc_macro, auc_weighted = None, None

    print(f"[ {model_id} ]")
    print(f"Accuracy -> {accuracy:.4f}")
    print(f"F1 Score -> (macro: {f1_macro:.4f}) | (weighted: {f1_weighted:.4f})")
    if auc_macro and auc_weighted:
        print(f"AUROC -> (macro: {auc_macro:.4f}) | (weighted: {auc_weighted:.4f})")
    else:
        print("AUROC -> X")
    print()


# %% 모델별 평가
import warnings

warnings.filterwarnings("ignore")

train_set = (X_train, y_train)
test_set = (X_test, y_test)
for model_id, model_instance in model_dict.items():
    eval_with_classifier(model_id, model_instance, train_set, test_set)
