import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import AdamW

from tqdm import tqdm

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


# Function for evaluation
def eval_model(model, dataloader, criterion):
    # Local vars
    cum_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    # Eval loop
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # Batch data -> GPU
            labels = batch.pop("labels").to("cuda")
            input_dicts = {k: v.to("cuda") for k, v in batch.items()}

            # Inferencing
            logits = model(**input_dicts)
            loss = criterion(logits, labels)
            cum_loss += loss.item()

            preds = torch.argmax(logits.detach(), dim=1)
            probs = F.softmax(logits, dim=1)

            # Extend output-related data list, respectively
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Metrics
    val_loss = cum_loss / len(dataloader)
    val_acc = accuracy_score(all_labels, all_preds)

    f1_macro = f1_score(all_labels, all_preds, average="macro")
    f1_weighted = f1_score(all_labels, all_preds, average="weighted")
    val_f1 = f1_macro if f1_macro > f1_weighted else f1_weighted

    try:
        auc_marco = roc_auc_score(
            all_labels, all_probs, multi_class="ovr", average="macro"
        )
        auc_weighted = roc_auc_score(
            all_labels, all_probs, multi_class="ovr", average="weighted"
        )
    except ValueError:
        auc_marco, auc_weighted = None, None
    val_auroc = auc_marco if auc_marco > auc_marco else auc_weighted

    return val_loss, val_acc, val_f1, val_auroc


# Function for train loop
def train_model(model, num_epochs, train_loader, valid_loader):
    # Local vars
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    # Train loop
    for epoch in range(num_epochs):
        model.train()
        cum_loss = 0.0
        correct_sp = 0
        total_sp = 0

        for idx, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            # Batch data -> GPU
            labels = batch.pop("labels").to("cuda")
            input_dicts = {k: v.to("cuda") for k, v in batch.items()}

            # Reset gradient
            optimizer.zero_grad()

            # Backpropagtion
            logits = model(**input_dicts)
            loss = criterion(logits, labels)
            loss.backward()

            optimizer.step()

            # Logging
            cum_loss += loss.item()
            predicted = torch.argmax(logits.detach(), dim=1)
            total_sp += labels.size(0)
            correct_sp += (predicted == labels).sum().item()

            if (idx + 1) % 10 == 0:
                avg_loss = cum_loss / (idx + 1)
                acc = 100.0 * correct_sp / total_sp
                print()
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Step [{idx + 1}/{len(train_loader)}], Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}%"
                )
        print()

        # Epoch results
        train_loss = cum_loss / len(train_loader)
        train_acc = 100.0 * correct_sp / total_sp

        val_loss, val_acc, val_f1, val_auroc = eval_model(
            model, dataloader=valid_loader, criterion=criterion
        )

        print(f"Epoch [{epoch + 1}/{num_epochs}] Completed!")
        print(f"Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
        print(
            f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, F1 Score: {val_f1:.4f}, AUROC: {val_auroc:.4f}"
        )
        print()

    return model


# Function to get embeddings from fine-tuned embedding models
def get_embeddings(embedding_model, dataloader):
    # Local vars
    embedding_list = []
    label_list = []

    # Embedding loop
    for batch in tqdm(dataloader):
        # Batch data -> GPU
        labels = batch.pop("labels").to("cuda")
        input_dicts = {k: v.to("cuda") for k, v in batch.items()}

        # Embedding
        with torch.no_grad():
            embedded_output = embedding_model(**input_dicts)
            batch_embeddings = embedded_output.pooler_output.float().cpu().numpy()
            batch_labels = labels.cpu().numpy()
            embedding_list.extend(batch_embeddings)
            label_list.extend(batch_labels)

    return np.array(embedding_list), np.array(label_list)


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

    print()
    print(f"[ {model_id} ]")
    print(f"Accuracy -> {accuracy:.4f}")
    print(f"F1 Score -> (macro: {f1_macro:.4f}) | (weighted: {f1_weighted:.4f})")
    if auc_macro and auc_weighted:
        print(f"AUROC -> (macro: {auc_macro:.4f}) | (weighted: {auc_weighted:.4f})")
    else:
        print("AUROC -> X")
    print()
