import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Any, Optional, Union
from ...core.base_classes import BaseClassificationModel
from ..utils.preprocess import preprocess_features
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class SimpleGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleGCN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x, adj):
        h = torch.relu(self.fc1(torch.matmul(adj, x)))
        out = self.fc2(torch.matmul(adj, h))
        return out

class GNNModel(BaseClassificationModel):
    """
    Simple Graph Neural Network (GCN) classification model using PyTorch.
    Implements the BaseClassificationModel interface.
    Expects features as node features and adjacency matrix in kwargs.
    """
    def __init__(self, input_dim=10, hidden_dim=32, output_dim=2, lr=0.001, epochs=20, device=None):
        super().__init__(
            name="gnn",
            description="Graph Convolutional Network (GCN) classifier for gait data classification"
        )
        self.config = {
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'output_dim': output_dim,
            'lr': lr,
            'epochs': epochs
        }
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleGCN(input_dim, hidden_dim, output_dim).to(self.device)
        self.epochs = epochs
        self.trained = False
        self.feature_names = []
        self.class_names = []

    def train(self, features: List[Dict], **kwargs):
        X, y = preprocess_features(features)
        # X: (num_nodes, num_features), y: (num_nodes,)
        adj = kwargs.get('adjacency_matrix')
        if adj is None:
            raise ValueError("Adjacency matrix must be provided as 'adjacency_matrix' in kwargs for GNN training.")
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.long).to(self.device)
        adj = torch.tensor(adj, dtype=torch.float32).to(self.device)
        self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        self.class_names = list(set(y.cpu().numpy()))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config['lr'])
        for epoch in range(self.epochs):
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(X, adj)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            if (epoch+1) % 5 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}")
        self.trained = True
        print("GNN model trained successfully.")

    def predict(self, features: List[Dict], **kwargs) -> np.ndarray:
        if not self.trained:
            raise ValueError("Model must be trained before making predictions")
        X, _ = preprocess_features(features)
        adj = kwargs.get('adjacency_matrix')
        if adj is None:
            raise ValueError("Adjacency matrix must be provided as 'adjacency_matrix' in kwargs for GNN prediction.")
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        adj = torch.tensor(adj, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X, adj)
            _, predicted = torch.max(outputs.data, 1)
        return predicted.cpu().numpy()

    def evaluate(self, features: List[Dict], **kwargs) -> Dict[str, float]:
        if not self.trained:
            raise ValueError("Model must be trained before evaluation")
        X, y = preprocess_features(features)
        adj = kwargs.get('adjacency_matrix')
        if adj is None:
            raise ValueError("Adjacency matrix must be provided as 'adjacency_matrix' in kwargs for GNN evaluation.")
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = np.array(y)
        adj = torch.tensor(adj, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X, adj)
            _, y_pred = torch.max(outputs.data, 1)
        y_pred = y_pred.cpu().numpy()
        accuracy = accuracy_score(y, y_pred)
        conf_matrix = confusion_matrix(y, y_pred)
        metrics = {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix.tolist()
        }
        detailed_report = kwargs.get('detailed_report', False)
        if detailed_report:
            class_report = classification_report(y, y_pred, output_dict=True)
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
        print(f"GNN model saved to {filepath}")

    def load_model(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model = SimpleGCN(
            self.config['input_dim'],
            self.config['hidden_dim'],
            self.config['output_dim']
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.config = checkpoint.get('config', self.config)
        self.feature_names = checkpoint.get('feature_names', [])
        self.class_names = checkpoint.get('class_names', [])
        self.trained = checkpoint.get('trained', True)
        print(f"GNN model loaded from {filepath}")
