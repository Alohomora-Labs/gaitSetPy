"""
Base classes for GaitSetPy components.

This module defines abstract base classes that all components should inherit from.
Each base class defines the interface and common functionality for its respective component type.

Maintainer: @aharshit123456
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import pandas as pd
import numpy as np
import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


class BaseDatasetLoader(ABC):
    """
    Base class for all dataset loaders.
    
    All dataset loaders should inherit from this class and implement the required methods.
    This class provides thread-safe concurrent downloading capabilities for efficient data retrieval.
    """
    
    def __init__(self, name: str, description: str = "", max_workers: int = 8):
        """
        Initialize the dataset loader.
        
        Args:
            name: Name of the dataset
            description: Description of the dataset
            max_workers: Maximum number of concurrent download threads (default: 8)
        """
        self.name = name
        self.description = description
        self.data = None
        self.metadata = {}
        self.max_workers = max_workers
        self._download_stats = {'success': 0, 'failed': 0, 'skipped': 0}
    
    @abstractmethod
    def load_data(self, data_dir: str, **kwargs) -> Tuple[List[pd.DataFrame], List[str]]:
        """
        Load dataset from the specified directory.
        
        Args:
            data_dir: Directory containing the dataset
            **kwargs: Additional arguments specific to the dataset
            
        Returns:
            Tuple of (data_list, names_list)
        """
        pass
    
    @abstractmethod
    def create_sliding_windows(self, data: List[pd.DataFrame], names: List[str], 
                             window_size: int = 192, step_size: int = 32) -> List[Dict]:
        """
        Create sliding windows from the loaded data.
        
        Args:
            data: List of DataFrames
            names: List of names corresponding to the data
            window_size: Size of each sliding window
            step_size: Step size for sliding windows
            
        Returns:
            List of dictionaries containing sliding windows
        """
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported file formats.
        
        Returns:
            List of supported file extensions
        """
        pass
    
    def _download_file(self, url: str, dest_path: str, 
                      chunk_size: int = 8192, timeout: int = 30) -> Tuple[bool, str]:
        """
        Download a single file from URL to destination path.
        
        This method is thread-safe and can be called concurrently.
        
        Args:
            url: URL to download from
            dest_path: Destination file path
            chunk_size: Size of chunks to download (default: 8192 bytes)
            timeout: Request timeout in seconds (default: 30)
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            # Check if file already exists
            if os.path.exists(dest_path):
                self._download_stats['skipped'] += 1
                return True, f"File already exists: {dest_path}"
            
            # Make the request
            response = requests.get(url, stream=True, timeout=timeout)
            
            if response.status_code == 200:
                # Ensure parent directory exists
                os.makedirs(os.path.dirname(dest_path) if os.path.dirname(dest_path) else '.', exist_ok=True)
                
                # Write file in chunks
                with open(dest_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                
                self._download_stats['success'] += 1
                return True, f"Successfully downloaded: {dest_path}"
            else:
                self._download_stats['failed'] += 1
                return False, f"HTTP {response.status_code}: {url}"
                
        except requests.exceptions.Timeout:
            self._download_stats['failed'] += 1
            return False, f"Timeout downloading: {url}"
        except requests.exceptions.RequestException as e:
            self._download_stats['failed'] += 1
            return False, f"Request error for {url}: {str(e)}"
        except IOError as e:
            self._download_stats['failed'] += 1
            return False, f"IO error for {dest_path}: {str(e)}"
        except Exception as e:
            self._download_stats['failed'] += 1
            return False, f"Unexpected error for {url}: {str(e)}"
    
    def download_files_concurrent(self, 
                                  download_tasks: List[Dict[str, str]], 
                                  show_progress: bool = True,
                                  desc: str = "Downloading files") -> Dict[str, Any]:
        """
        Download multiple files concurrently using a thread pool.
        
        Args:
            download_tasks: List of dicts with 'url' and 'dest_path' keys
            show_progress: Whether to show progress bar (default: True)
            desc: Description for progress bar
            
        Returns:
            Dictionary with download statistics and results
            
        Example:
            tasks = [
                {'url': 'http://example.com/file1.txt', 'dest_path': '/path/to/file1.txt'},
                {'url': 'http://example.com/file2.txt', 'dest_path': '/path/to/file2.txt'}
            ]
            results = loader.download_files_concurrent(tasks)
        """
        # Reset stats
        self._download_stats = {'success': 0, 'failed': 0, 'skipped': 0}
        
        results = []
        failed_downloads = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all download tasks
            future_to_task = {
                executor.submit(self._download_file, task['url'], task['dest_path']): task
                for task in download_tasks
            }
            
            # Process completed tasks with optional progress bar
            if show_progress:
                futures = tqdm(as_completed(future_to_task), 
                             total=len(download_tasks), 
                             desc=desc)
            else:
                futures = as_completed(future_to_task)
            
            for future in futures:
                task = future_to_task[future]
                try:
                    success, message = future.result()
                    results.append({
                        'url': task['url'],
                        'dest_path': task['dest_path'],
                        'success': success,
                        'message': message
                    })
                    
                    if not success:
                        failed_downloads.append({
                            'url': task['url'],
                            'dest_path': task['dest_path'],
                            'error': message
                        })
                        
                except Exception as e:
                    error_msg = f"Exception during download: {str(e)}"
                    results.append({
                        'url': task['url'],
                        'dest_path': task['dest_path'],
                        'success': False,
                        'message': error_msg
                    })
                    failed_downloads.append({
                        'url': task['url'],
                        'dest_path': task['dest_path'],
                        'error': error_msg
                    })
        
        # Return comprehensive results
        return {
            'total': len(download_tasks),
            'success': self._download_stats['success'],
            'failed': self._download_stats['failed'],
            'skipped': self._download_stats['skipped'],
            'failed_downloads': failed_downloads,
            'all_results': results
        }
    
    def set_max_workers(self, max_workers: int):
        """
        Set the maximum number of concurrent download threads.
        
        Args:
            max_workers: Maximum number of threads (must be positive)
        """
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        self.max_workers = max_workers
    
    def get_download_stats(self) -> Dict[str, int]:
        """
        Get statistics from the last download operation.
        
        Returns:
            Dictionary with success, failed, and skipped counts
        """
        return self._download_stats.copy()
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the dataset.
        
        Returns:
            Dictionary containing dataset information
        """
        return {
            'name': self.name,
            'description': self.description,
            'metadata': self.metadata,
            'supported_formats': self.get_supported_formats(),
            'max_workers': self.max_workers
        }


class BaseFeatureExtractor(ABC):
    """
    Base class for all feature extractors.
    
    All feature extractors should inherit from this class and implement the required methods.
    """
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize the feature extractor.
        
        Args:
            name: Name of the feature extractor
            description: Description of the feature extractor
        """
        self.name = name
        self.description = description
        self.config = {}
    
    @abstractmethod
    def extract_features(self, windows: List[Dict], fs: int, **kwargs) -> List[Dict]:
        """
        Extract features from sliding windows.
        
        Args:
            windows: List of sliding window dictionaries
            fs: Sampling frequency
            **kwargs: Additional arguments for feature extraction
            
        Returns:
            List of feature dictionaries
        """
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """
        Get names of features extracted by this extractor.
        
        Returns:
            List of feature names
        """
        pass
    
    def configure(self, config: Dict[str, Any]):
        """
        Configure the feature extractor.
        
        Args:
            config: Configuration dictionary
        """
        self.config.update(config)
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the feature extractor.
        
        Returns:
            Dictionary containing feature extractor information
        """
        return {
            'name': self.name,
            'description': self.description,
            'config': self.config,
            'feature_names': self.get_feature_names()
        }


class BasePreprocessor(ABC):
    """
    Base class for all preprocessors.
    
    All preprocessors should inherit from this class and implement the required methods.
    """
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize the preprocessor.
        
        Args:
            name: Name of the preprocessor
            description: Description of the preprocessor
        """
        self.name = name
        self.description = description
        self.config = {}
        self.fitted = False
    
    @abstractmethod
    def fit(self, data: Union[pd.DataFrame, np.ndarray], **kwargs):
        """
        Fit the preprocessor to the data.
        
        Args:
            data: Input data to fit on
            **kwargs: Additional arguments for fitting
        """
        pass
    
    @abstractmethod
    def transform(self, data: Union[pd.DataFrame, np.ndarray], **kwargs) -> Union[pd.DataFrame, np.ndarray]:
        """
        Transform the data using the fitted preprocessor.
        
        Args:
            data: Input data to transform
            **kwargs: Additional arguments for transformation
            
        Returns:
            Transformed data
        """
        pass
    
    def fit_transform(self, data: Union[pd.DataFrame, np.ndarray], **kwargs) -> Union[pd.DataFrame, np.ndarray]:
        """
        Fit the preprocessor and transform the data.
        
        Args:
            data: Input data to fit and transform
            **kwargs: Additional arguments
            
        Returns:
            Transformed data
        """
        self.fit(data, **kwargs)
        return self.transform(data, **kwargs)
    
    def configure(self, config: Dict[str, Any]):
        """
        Configure the preprocessor.
        
        Args:
            config: Configuration dictionary
        """
        self.config.update(config)
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the preprocessor.
        
        Returns:
            Dictionary containing preprocessor information
        """
        return {
            'name': self.name,
            'description': self.description,
            'config': self.config,
            'fitted': self.fitted
        }


class BaseEDAAnalyzer(ABC):
    """
    Base class for all EDA analyzers.
    
    All EDA analyzers should inherit from this class and implement the required methods.
    """
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize the EDA analyzer.
        
        Args:
            name: Name of the EDA analyzer
            description: Description of the EDA analyzer
        """
        self.name = name
        self.description = description
        self.config = {}
    
    @abstractmethod
    def analyze(self, data: Union[pd.DataFrame, List[pd.DataFrame]], **kwargs) -> Dict[str, Any]:
        """
        Perform analysis on the data.
        
        Args:
            data: Input data to analyze
            **kwargs: Additional arguments for analysis
            
        Returns:
            Dictionary containing analysis results
        """
        pass
    
    @abstractmethod
    def visualize(self, data: Union[pd.DataFrame, List[pd.DataFrame]], **kwargs):
        """
        Create visualizations of the data.
        
        Args:
            data: Input data to visualize
            **kwargs: Additional arguments for visualization
        """
        pass
    
    def configure(self, config: Dict[str, Any]):
        """
        Configure the EDA analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config.update(config)
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the EDA analyzer.
        
        Returns:
            Dictionary containing EDA analyzer information
        """
        return {
            'name': self.name,
            'description': self.description,
            'config': self.config
        }


class BaseClassificationModel(ABC):
    """
    Base class for all classification models.
    
    All classification models should inherit from this class and implement the required methods.
    """
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize the classification model.
        
        Args:
            name: Name of the classification model
            description: Description of the classification model
        """
        self.name = name
        self.description = description
        self.model = None
        self.config = {}
        self.trained = False
    
    @abstractmethod
    def train(self, features: List[Dict], **kwargs):
        """
        Train the classification model.
        
        Args:
            features: List of feature dictionaries
            **kwargs: Additional arguments for training
        """
        pass
    
    @abstractmethod
    def predict(self, features: List[Dict], **kwargs) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            features: List of feature dictionaries
            **kwargs: Additional arguments for prediction
            
        Returns:
            Array of predictions
        """
        pass
    
    @abstractmethod
    def evaluate(self, features: List[Dict], **kwargs) -> Dict[str, float]:
        """
        Evaluate the model performance.
        
        Args:
            features: List of feature dictionaries
            **kwargs: Additional arguments for evaluation
            
        Returns:
            Dictionary containing evaluation metrics
        """
        pass
    
    @abstractmethod
    def save_model(self, filepath: str):
        """
        Save the trained model to a file.
        
        Args:
            filepath: Path to save the model
        """
        pass
    
    @abstractmethod
    def load_model(self, filepath: str):
        """
        Load a trained model from a file.
        
        Args:
            filepath: Path to the saved model
        """
        pass
    
    def configure(self, config: Dict[str, Any]):
        """
        Configure the classification model.
        
        Args:
            config: Configuration dictionary
        """
        self.config.update(config)
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the classification model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'name': self.name,
            'description': self.description,
            'config': self.config,
            'trained': self.trained
        } 