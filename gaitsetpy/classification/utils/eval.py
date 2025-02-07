'''
For evaluation of a classification model

Maintainer: @aharshit123456
'''

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from .preprocess import preprocess_features

def evaluate_model(model, features):
    """
    Evaluates the given model on the provided features and prints accuracy and confusion matrix.
    """
    X, y = preprocess_features(features)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    # conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")
    # print(f"Confusion Matrix:\n{conf_matrix}")