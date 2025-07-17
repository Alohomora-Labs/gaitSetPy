from typing import List, Dict, Any, Optional, Union
import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from ...core.base_classes import BaseClassificationModel
from ..utils.preprocess import preprocess_features

class MLPModel(BaseClassificationModel):
    """
    Multi-Layer Perceptron (MLP) classification model.
    Implements the BaseClassificationModel interface using scikit-learn's MLPClassifier.
    """
    def __init__(self, hidden_layer_sizes=(100,), activation='relu', solver='adam', random_state=42, max_iter=200):
        super().__init__(
            name="mlp",
            description="Multi-Layer Perceptron classifier for gait data classification"
        )
        self.config = {
            'hidden_layer_sizes': hidden_layer_sizes,
            'activation': activation,
            'solver': solver,
            'random_state': random_state,
            'max_iter': max_iter
        }
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            random_state=random_state,
            max_iter=max_iter
        )
        self.feature_names = []
        self.class_names = []

    def train(self, features: List[Dict], **kwargs):
        X, y = preprocess_features(features)
        self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        self.class_names = list(set(y))
        test_size = kwargs.get('test_size', 0.2)
        validation_split = kwargs.get('validation_split', True)
        if validation_split:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.config['random_state']
            )
            self.model.fit(X_train, y_train)
            self.X_test = X_test
            self.y_test = y_test
            train_accuracy = self.model.score(X_train, y_train)
            test_accuracy = self.model.score(X_test, y_test)
            print(f"Training accuracy: {train_accuracy:.4f}")
            print(f"Validation accuracy: {test_accuracy:.4f}")
        else:
            self.model.fit(X, y)
            train_accuracy = self.model.score(X, y)
            print(f"Training accuracy: {train_accuracy:.4f}")
        self.trained = True
        print("MLP model trained successfully.")

    def predict(self, features: List[Dict], **kwargs) -> Union[np.ndarray, Any]:
        if not self.trained:
            raise ValueError("Model must be trained before making predictions")
        X, _ = preprocess_features(features)
        return_probabilities = kwargs.get('return_probabilities', False)
        if return_probabilities:
            return self.model.predict_proba(X)
        else:
            return self.model.predict(X)

    def evaluate(self, features: List[Dict], **kwargs) -> Dict[str, float]:
        if not self.trained:
            raise ValueError("Model must be trained before evaluation")
        if hasattr(self, 'X_test') and hasattr(self, 'y_test'):
            X_test, y_test = self.X_test, self.y_test
        else:
            X_test, y_test = preprocess_features(features)
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        metrics = {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix.tolist()
        }
        detailed_report = kwargs.get('detailed_report', False)
        if detailed_report:
            class_report = classification_report(y_test, y_pred, output_dict=True)
            metrics['classification_report'] = class_report
        return metrics

    def save_model(self, filepath: str):
        if not self.trained:
            raise ValueError("Model must be trained before saving")
        model_data = {
            'model': self.model,
            'config': self.config,
            'feature_names': self.feature_names,
            'class_names': self.class_names,
            'trained': self.trained
        }
        joblib.dump(model_data, filepath)
        print(f"MLP model saved to {filepath}")

    def load_model(self, filepath: str):
        try:
            model_data = joblib.load(filepath)
            if isinstance(model_data, dict):
                self.model = model_data['model']
                self.config = model_data.get('config', self.config)
                self.feature_names = model_data.get('feature_names', [])
                self.class_names = model_data.get('class_names', [])
                self.trained = model_data.get('trained', False)
            else:
                self.model = model_data
                self.trained = True
            print(f"MLP model loaded from {filepath}")
        except Exception as e:
            print(f"Failed to load MLP model: {e}")
