from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
):
    model.fit(X_train, y_train)

    val_predictions = model.predict(X_val)
    test_predictions = model.predict(X_test)

    val_accuracy = accuracy_score(y_val, val_predictions)
    val_precision = precision_score(y_val, val_predictions)
    val_recall = recall_score(y_val, val_predictions)
    val_f1 = f1_score(y_val, val_predictions)

    test_accuracy = accuracy_score(y_test, test_predictions)
    test_precision = precision_score(y_test, test_predictions)
    test_recall = recall_score(y_test, test_predictions)
    test_f1 = f1_score(y_test, test_predictions)

    print("Validation:")
    print(f"Accuracy:  {val_accuracy:.4f}")
    print(f"Precision: {val_precision:.4f}")
    print(f"Recall:    {val_recall:.4f}")
    print(f"F1:        {val_f1:.4f}")

    print("\nTest:")
    print(f"Accuracy:  {test_accuracy:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall:    {test_recall:.4f}")
    print(f"F1:        {test_f1:.4f}")

    print("\nValidation classification report:")
    print(classification_report(y_val, val_predictions))

    print("\nTest classification report:")
    print(classification_report(y_test, test_predictions))

    print("\nTest confusion matrix:")
    print(confusion_matrix(y_test, test_predictions))

    return model