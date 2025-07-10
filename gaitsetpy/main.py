"""
GaitSetPy Main Module

This module provides the main entry point and example usage of the GaitSetPy library.
"""

import os
from typing import Dict, Any, List, Optional
from .dataset.daphnet import DaphnetDataset
from .preprocessing.pipeline import PreprocessingPipeline
from .features.gait_features import FeatureExtractor
from .classification.models.random_forest import RandomForestModel
from .eda.visualization import plot_sensor_data, plot_features

def process_gait_data(
    data_dir: str,
    window_size: int = 192,
    overlap: float = 0.5,
    fs: int = 100,
    **kwargs
) -> Dict[str, Any]:
    """
    Process gait data using the GaitSetPy pipeline.
    
    Args:
        data_dir (str): Directory containing the dataset
        window_size (int): Size of sliding windows
        overlap (float): Overlap between windows (0-1)
        fs (int): Sampling frequency
        **kwargs: Additional parameters for preprocessing and feature extraction
        
    Returns:
        Dict[str, Any]: Dictionary containing processed data and results
    """
    # Initialize components
    dataset = DaphnetDataset(data_dir)
    preprocessor = PreprocessingPipeline()
    feature_extractor = FeatureExtractor()
    model = RandomForestModel()
    
    # Load and preprocess data
    print("Loading dataset...")
    data, subject_ids = dataset.load()
    
    # Process each subject's data
    all_features = []
    for i, subject_data in enumerate(data):
        print(f"Processing subject {subject_ids[i]}...")
        
        # Preprocess data
        processed_data = preprocessor.process(subject_data)
        
        # Create windows
        windows = dataset.create_sliding_windows(processed_data, window_size, overlap)
        
        # Extract features for each window
        for window in windows:
            features = feature_extractor.extract(window, fs=fs)
            features["subject_id"] = subject_ids[i]
            all_features.append(features)
    
    # Train model
    print("Training model...")
    model.train(all_features)
    
    # Save results
    results = {
        "features": all_features,
        "model": model,
        "subject_ids": subject_ids,
        "config": {
            "window_size": window_size,
            "overlap": overlap,
            "fs": fs,
            **kwargs
        }
    }
    
    return results

def main():
    """Main entry point."""
    # Example usage
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Process data
    results = process_gait_data(
        data_dir=data_dir,
        window_size=192,
        overlap=0.5,
        fs=100,
        time_domain=True,
        frequency_domain=True,
        statistical=True
    )
    
    # Save model
    model_path = os.path.join(data_dir, "random_forest_model.pkl")
    results["model"].save(model_path)
    print(f"Model saved to {model_path}")
    
    # Plot results
    plot_sensor_data(results["features"])
    plot_features(results["features"])

if __name__ == "__main__":
    main()

