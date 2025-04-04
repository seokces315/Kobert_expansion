# %% Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    f1_score,
    confusion_matrix,
)
from sklearn.feature_extraction.text import TfidfVectorizer
import random
import re


# %% Seed
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


set_seed(42)

# %% 데이터 로드
df = pd.read_csv("KoreanData.csv")
df = df[df["input_text_long"].notnull()]
df = df[df["answer_rate"].notnull()]
df["answer_rate"] = df["answer_rate"].astype(float)


# %% 텍스트 전처리
def preprocess(text):
    text = re.sub(r"[^\w\s]", "", text)
    return text


texts = [preprocess(t) for t in df["input_text_long"].tolist()]

# %% TF-IDF 벡터화
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

# %% 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    embeddings, df["label"].values, test_size=0.2, random_state=42, stratify=df["label"]
)
print(X_train.shape, y_train.shape)

# %% 모델 리스트


models = {
    "LightGBM": LGBMClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=7, random_state=42, verbose=-1
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=500, max_depth=10, random_state=42, n_jobs=-1
    ),
    "LogisticRegression": LogisticRegression(max_iter=1000, multi_class="ovr"),
    "SVC": SVC(C=1.0, probability=True),
    "CatBoost": CatBoostClassifier(
        iterations=500, learning_rate=0.05, depth=7, verbose=0, random_state=42
    ),
}


# %% 평가 함수
def evaluate_classifier(name, model):
    model.fit(X_train, y_train)
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    test_probs = model.predict_proba(X_test)

    acc = accuracy_score(y_test, test_preds)
    f1_macro = f1_score(y_test, test_preds, average="macro")
    f1_weighted = f1_score(y_test, test_preds, average="weighted")
    try:
        auroc_macro = roc_auc_score(
            y_test, test_probs, multi_class="ovr", average="macro"
        )
        auroc_weighted = roc_auc_score(
            y_test, test_probs, multi_class="ovr", average="weighted"
        )
    except ValueError:
        auroc_macro, auroc_weighted = None, None

    print(f"\n🚀 {name} 분류 결과")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score (macro): {f1_macro:.4f} | (weighted): {f1_weighted:.4f}")
    if auroc_macro:
        print(f"AUROC (macro): {auroc_macro:.4f} | (weighted): {auroc_weighted:.4f}")
    else:
        print(f"⚠️ AUROC 계산 불가 (클래스 결측)")

    print("\n📊 Classification Report:")
    print(classification_report(y_test, test_preds, digits=4))
    print("📌 Confusion Matrix:")
    print(confusion_matrix(y_test, test_preds))


# %% 모델별 평가
for name, model in models.items():
    evaluate_classifier(name, model)

# %%
