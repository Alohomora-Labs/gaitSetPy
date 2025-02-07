'''
This module contains the RandomForestModel class which is used to train and predict using a Random Forest model.

Maintainer: @aharshit123456
'''

from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np
from ..utils.preprocess import preprocess_features

class RandomForestModel:
    def __init__(self, n_estimators=100, random_state=42):
        """
        Initializes the RandomForestModel.
        """
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    def preprocess_features(self, features):
        """
        Calls the preprocess_features function from the utils module.
        """
        return preprocess_features(features)

    def train(self, features):
        """
        Trains the Random Forest model on the given features.
        """
        X, y = self.preprocess_features(features)
        self.model.fit(X, y)
        print("Model trained successfully.")

    def predict(self, features):
        """
        Predicts labels for given feature inputs.
        """
        X, _ = self.preprocess_features(features)
        return self.model.predict(X)

    def load_pretrained_weights(self, filepath):
        """
        Loads pre-trained model weights from a file.
        """
        if filepath == "random_forest_model.pkl":
            filepath = "../weights/random_forest_model.pkl"

        self.model = joblib.load(filepath)
        print("Pretrained model loaded successfully.")

    def save_model(self, filepath):
        """
        Saves the trained model to a file.
        """
        joblib.dump(self.model, filepath)
        print("Model saved successfully.")
