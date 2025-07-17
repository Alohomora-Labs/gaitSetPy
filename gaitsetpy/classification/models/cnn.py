import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Any, Optional, Union
from ...core.base_classes import BaseClassificationModel
from ..utils.preprocess import preprocess_features
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class SimpleCNN(nn.Module):
    def __init__(self, input_channels, num_classes, seq_len=1):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(32, num_classes)
    def forward(self, x):
        # x: (batch, channels, seq_len)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class CNNModel(BaseClassificationModel):
    """
    Simple 1D CNN classification model using PyTorch.
    Implements the BaseClassificationModel interface.
    """
    def __init__(self, input_channels=10, num_classes=2, lr=0.001, epochs=20, batch_size=32, device=None):
        super().__init__(
            name="cnn",
            description="1D CNN classifier for gait data classification"
        )
        self.config = {
            'input_channels': input_channels,
            'num_classes': num_classes,
            'lr': lr,
            'epochs': epochs,
            'batch_size': batch_size
        }
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleCNN(input_channels, num_classes).to(self.device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.trained = False
        self.feature_names = []
        self.class_names = []

    def train(self, features: List[Dict], **kwargs):
        X, y = preprocess_features(features)
        # Reshape X for CNN: (samples, channels, seq_len)
        # Here, treat each feature vector as a channel with seq_len=1
        X = X.reshape((X.shape[0], X.shape[1], 1))
        self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        self.class_names = list(set(y))
        test_size = kwargs.get('test_size', 0.2)
        validation_split = kwargs.get('validation_split', True)
        if validation_split:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            self.X_test = X_test
            self.y_test = y_test
        else:
            X_train, y_train = X, y
        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.long).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config['lr'])
        for epoch in range(self.epochs):
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            if (epoch+1) % 5 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}")
        self.trained = True
        print("CNN model trained successfully.")

    def predict(self, features: List[Dict], **kwargs) -> np.ndarray:
        if not self.trained:
            raise ValueError("Model must be trained before making predictions")
        X, _ = preprocess_features(features)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            _, predicted = torch.max(outputs.data, 1)
        return predicted.cpu().numpy()

    def evaluate(self, features: List[Dict], **kwargs) -> Dict[str, float]:
        if not self.trained:
            raise ValueError("Model must be trained before evaluation")
        if hasattr(self, 'X_test') and hasattr(self, 'y_test'):
            X_test, y_test = self.X_test, self.y_test
        else:
            X_test, y_test = preprocess_features(features)
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        y_test = np.array(y_test)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_test)
            _, y_pred = torch.max(outputs.data, 1)
        y_pred = y_pred.cpu().numpy()
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
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'feature_names': self.feature_names,
            'class_names': self.class_names,
            'trained': self.trained
        }, filepath)
        print(f"CNN model saved to {filepath}")

    def load_model(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model = SimpleCNN(
            self.config['input_channels'],
            self.config['num_classes']
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.config = checkpoint.get('config', self.config)
        self.feature_names = checkpoint.get('feature_names', [])
        self.class_names = checkpoint.get('class_names', [])
        self.trained = checkpoint.get('trained', True)
        print(f"CNN model loaded from {filepath}") 