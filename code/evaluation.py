from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)


# Function to evaluate embeddings with various classifiers
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
