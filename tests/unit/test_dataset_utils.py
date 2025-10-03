"""
Unit tests for dataset utility functions in GaitSetPy.

This module tests the dataset download, extraction, and utility functions
in the dataset.utils module.

Maintainer: @aharshit123456
"""

import pytest
import os
import tempfile
import zipfile
import tarfile
from unittest.mock import patch, Mock, MagicMock, mock_open
import requests
from concurrent.futures import ThreadPoolExecutor

from gaitsetpy.dataset.utils import (
    download_dataset,
    download_daphnet_data,
    download_mobifall_data,
    download_arduous_data,
    download_urfall_data,
    download_harup_data,
    extract_dataset,
    extract_daphnet_data,
    extract_mobifall_data,
    extract_arduous_data,
    extract_urfall_data,
    extract_harup_data,
    sliding_window,
    _download_file
)


class TestDownloadDataset:
    """Test cases for the main download_dataset function."""
    
    def test_download_daphnet_dataset(self):
        """Test downloading Daphnet dataset."""
        with patch('gaitsetpy.dataset.utils.download_daphnet_data') as mock_download:
            download_dataset("daphnet", "/test/data")
            mock_download.assert_called_once_with("/test/data")
    
    def test_download_mobifall_dataset(self):
        """Test downloading MobiFall dataset."""
        with patch('gaitsetpy.dataset.utils.download_mobifall_data') as mock_download:
            download_dataset("mobifall", "/test/data")
            mock_download.assert_called_once_with("/test/data")
    
    def test_download_arduous_dataset(self):
        """Test downloading Arduous dataset."""
        with patch('gaitsetpy.dataset.utils.download_arduous_data') as mock_download:
            download_dataset("arduous", "/test/data")
            mock_download.assert_called_once_with("/test/data")
    
    def test_download_harup_dataset(self):
        """Test downloading HAR-UP dataset."""
        with patch('gaitsetpy.dataset.utils.download_harup_data') as mock_download:
            download_dataset("harup", "/test/data")
            mock_download.assert_called_once_with("/test/data")
    
    def test_download_urfall_dataset(self):
        """Test downloading UrFall dataset."""
        with patch('gaitsetpy.dataset.utils.download_urfall_data') as mock_download:
            download_dataset("urfall", "/test/data")
            mock_download.assert_called_once_with("/test/data")
    
    def test_download_physionet_dataset(self):
        """Test downloading PhysioNet dataset (should pass silently)."""
        # PhysioNet is handled by the loader itself, so this should not raise an error
        download_dataset("physionet", "/test/data")
    
    def test_download_unsupported_dataset(self):
        """Test downloading unsupported dataset."""
        with pytest.raises(ValueError, match="Dataset unsupported not supported"):
            download_dataset("unsupported", "/test/data")


class TestDownloadDaphnetData:
    """Test cases for Daphnet dataset download."""
    
    @patch('gaitsetpy.dataset.utils.requests.get')
    @patch('gaitsetpy.dataset.utils.requests.head')
    @patch('tqdm.tqdm')
    @patch('gaitsetpy.dataset.utils.os.makedirs')
    @patch('gaitsetpy.dataset.utils.os.path.exists')
    @patch('gaitsetpy.dataset.utils.os.path.getsize')
    @patch('builtins.open', new_callable=mock_open)
    def test_download_daphnet_success(self, mock_file, mock_getsize, mock_exists, 
                                    mock_makedirs, mock_tqdm, mock_head, mock_get):
        """Test successful Daphnet download."""
        # Mock file doesn't exist
        mock_exists.return_value = False
        mock_getsize.return_value = 1024
        
        # Mock HTTP responses
        mock_head_response = Mock()
        mock_head_response.headers = {'content-length': '1024'}
        mock_head.return_value = mock_head_response
        
        mock_get_response = Mock()
        mock_get_response.headers = {'content-length': '1024'}
        mock_get_response.iter_content.return_value = [b'chunk1', b'chunk2']
        mock_get_response.raise_for_status.return_value = None
        mock_get.return_value = mock_get_response
        
        # Mock tqdm
        mock_progress = Mock()
        mock_tqdm.return_value = mock_progress
        
        # Test download
        result = download_daphnet_data("/test/data")
        
        assert result.endswith("daphnet.zip")
        mock_makedirs.assert_called_once_with("/test/data", exist_ok=True)
        mock_get.assert_called_once()
        # Check that file was opened with correct mode, allowing for path separator differences
        mock_file.assert_called_once()
        call_args = mock_file.call_args
        assert call_args[0][1] == "wb"  # Check the mode parameter
        assert "daphnet.zip" in str(call_args[0][0])  # Check filename is in path
    
    @patch('gaitsetpy.dataset.utils.os.path.exists')
    @patch('gaitsetpy.dataset.utils.os.path.getsize')
    def test_download_daphnet_file_exists(self, mock_getsize, mock_exists):
        """Test Daphnet download when file already exists."""
        # Mock file exists and has content
        mock_exists.return_value = True
        mock_getsize.return_value = 1024
        
        result = download_daphnet_data("/test/data")
        
        assert result.endswith("daphnet.zip")
    
    @patch('gaitsetpy.dataset.utils.requests.get')
    @patch('gaitsetpy.dataset.utils.requests.head')
    @patch('gaitsetpy.dataset.utils.os.makedirs')
    @patch('gaitsetpy.dataset.utils.os.path.exists')
    @patch('gaitsetpy.dataset.utils.os.path.getsize')
    @patch('builtins.open', new_callable=mock_open)
    def test_download_daphnet_connection_error(self, mock_file, mock_getsize, mock_exists,
                                             mock_makedirs, mock_head, mock_get):
        """Test Daphnet download with connection error."""
        # Mock file doesn't exist
        mock_exists.return_value = False
        mock_getsize.return_value = 0
        
        # Mock HTTP responses
        mock_head_response = Mock()
        mock_head_response.headers = {'content-length': '1024'}
        mock_head.return_value = mock_head_response
        
        # Mock connection error
        mock_get.side_effect = requests.exceptions.RequestException("Connection failed")
        
        with pytest.raises(ConnectionError, match="Failed to download dataset"):
            download_daphnet_data("/test/data")
    
    @patch('gaitsetpy.dataset.utils.requests.get')
    @patch('gaitsetpy.dataset.utils.requests.head')
    @patch('gaitsetpy.dataset.utils.os.makedirs')
    @patch('gaitsetpy.dataset.utils.os.path.exists')
    @patch('gaitsetpy.dataset.utils.os.path.getsize')
    @patch('builtins.open', new_callable=mock_open)
    def test_download_daphnet_io_error(self, mock_file, mock_getsize, mock_exists,
                                     mock_makedirs, mock_head, mock_get):
        """Test Daphnet download with IO error."""
        # Mock file doesn't exist
        mock_exists.return_value = False
        mock_getsize.return_value = 0
        
        # Mock HTTP responses
        mock_head_response = Mock()
        mock_head_response.headers = {'content-length': '1024'}
        mock_head.return_value = mock_head_response
        
        mock_get_response = Mock()
        mock_get_response.headers = {'content-length': '1024'}
        mock_get_response.iter_content.return_value = [b'chunk1', b'chunk2']
        mock_get_response.raise_for_status.return_value = None
        mock_get.return_value = mock_get_response
        
        # Mock IO error
        mock_file.side_effect = IOError("Write failed")
        
        with pytest.raises(IOError, match="Failed to save dataset"):
            download_daphnet_data("/test/data")
    
    @patch('gaitsetpy.dataset.utils.requests.get')
    @patch('gaitsetpy.dataset.utils.requests.head')
    @patch('gaitsetpy.dataset.utils.os.makedirs')
    @patch('gaitsetpy.dataset.utils.os.path.exists')
    @patch('gaitsetpy.dataset.utils.os.path.getsize')
    @patch('builtins.open', new_callable=mock_open)
    def test_download_daphnet_unexpected_error(self, mock_file, mock_getsize, mock_exists,
                                             mock_makedirs, mock_head, mock_get):
        """Test Daphnet download with unexpected error."""
        # Mock file doesn't exist
        mock_exists.return_value = False
        mock_getsize.return_value = 0
        
        # Mock HTTP responses
        mock_head_response = Mock()
        mock_head_response.headers = {'content-length': '1024'}
        mock_head.return_value = mock_head_response
        
        # Mock unexpected error
        mock_get.side_effect = Exception("Unexpected error")
        
        with pytest.raises(Exception, match="Download failed"):
            download_daphnet_data("/test/data")


class TestDownloadUrfallData:
    """Test cases for UrFall dataset download."""
    
    @patch('gaitsetpy.dataset.utils.ThreadPoolExecutor')
    @patch('gaitsetpy.dataset.utils._download_file')
    @patch('gaitsetpy.dataset.utils.os.makedirs')
    @patch('gaitsetpy.dataset.utils.os.path.exists')
    def test_download_urfall_success(self, mock_exists, mock_makedirs, mock_download_file, mock_executor):
        """Test successful UrFall download."""
        # Mock file doesn't exist
        mock_exists.return_value = False
        
        # Mock ThreadPoolExecutor
        mock_executor_instance = Mock()
        mock_executor.return_value.__enter__.return_value = mock_executor_instance
        mock_executor.return_value.__exit__.return_value = None
        
        # Mock futures
        mock_future1 = Mock()
        mock_future1.result.return_value = (True, "/test/data/fall-01-acc.csv")
        mock_future2 = Mock()
        mock_future2.result.return_value = (True, "/test/data/adl-01-acc.csv")
        
        # Create a proper future mapping
        future_to_job = {mock_future1: ("url1", "dest1"), mock_future2: ("url2", "dest2")}
        
        mock_executor_instance.submit.return_value = mock_future1
        mock_executor_instance.__iter__ = Mock(return_value=iter([mock_future1, mock_future2]))
        
        # Mock as_completed
        with patch('gaitsetpy.dataset.utils.as_completed') as mock_as_completed:
            mock_as_completed.return_value = [mock_future1, mock_future2]
            
            # Mock the future_to_job dictionary access
            with patch('gaitsetpy.dataset.utils.future_to_job', future_to_job):
                result = download_urfall_data("/test/data", use_falls=True, use_adls=True, max_workers=4)
                
                assert result == "/test/data"
                mock_makedirs.assert_called_once_with("/test/data", exist_ok=True)
    
    @patch('gaitsetpy.dataset.utils.os.path.exists')
    def test_download_urfall_all_files_exist(self, mock_exists):
        """Test UrFall download when all files already exist."""
        # Mock all files exist
        mock_exists.return_value = True
        
        result = download_urfall_data("/test/data", use_falls=True, use_adls=True)
        
        assert result == "/test/data"
    
    @patch('gaitsetpy.dataset.utils.os.path.exists')
    def test_download_urfall_specific_sequences(self, mock_exists):
        """Test UrFall download with specific sequences."""
        # Mock file doesn't exist
        mock_exists.return_value = False
        
        with patch('gaitsetpy.dataset.utils.ThreadPoolExecutor') as mock_executor:
            mock_executor_instance = Mock()
            mock_executor.return_value.__enter__.return_value = mock_executor_instance
            mock_executor.return_value.__exit__.return_value = None
            
            mock_future = Mock()
            mock_future.result.return_value = (True, "/test/data/fall-01-acc.csv")
            mock_executor_instance.submit.return_value = mock_future
            mock_executor_instance.__iter__ = Mock(return_value=iter([mock_future]))
            
            with patch('gaitsetpy.dataset.utils.as_completed') as mock_as_completed:
                mock_as_completed.return_value = [mock_future]
                
                result = download_urfall_data(
                    "/test/data", 
                    sequences=['fall-01', 'adl-01'],
                    data_types=['accelerometer'],
                    max_workers=2
                )
                
                assert result == "/test/data"


class TestDownloadHarupData:
    """Test cases for HAR-UP dataset download."""
    
    @patch('gaitsetpy.dataset.utils.input')
    @patch('gaitsetpy.dataset.utils.os.path.exists')
    @patch('gaitsetpy.dataset.utils.os.makedirs')
    def test_download_harup_automatic_choice(self, mock_makedirs, mock_exists, mock_input):
        """Test HAR-UP download with automatic choice."""
        # Mock dataset doesn't exist
        mock_exists.return_value = False
        mock_input.return_value = "1"  # Automatic download choice
        
        with patch('gaitsetpy.dataset.utils.requests.get') as mock_get:
            with patch('tqdm.tqdm') as mock_tqdm:
                with patch('gaitsetpy.dataset.utils.zipfile.ZipFile') as mock_zip:
                    with patch('builtins.open', mock_open()) as mock_file:
                        # Mock successful download
                        mock_response = Mock()
                        mock_response.headers = {'content-length': '1024'}
                        mock_response.iter_content.return_value = [b'chunk1', b'chunk2']
                        mock_response.raise_for_status.return_value = None
                        mock_get.return_value = mock_response
                        
                        mock_progress = Mock()
                        mock_tqdm.return_value = mock_progress
                        
                        mock_zip_instance = Mock()
                        mock_zip.return_value.__enter__.return_value = mock_zip_instance
                        mock_zip.return_value.__exit__.return_value = None
                        
                        result = download_harup_data("/test/data")
                        
                        assert result.endswith("DataSet")
                        mock_makedirs.assert_called_once_with("/test/data", exist_ok=True)
    
    @patch('gaitsetpy.dataset.utils.input')
    @patch('gaitsetpy.dataset.utils.os.path.exists')
    def test_download_harup_manual_choice(self, mock_exists, mock_input):
        """Test HAR-UP download with manual choice."""
        # Mock dataset doesn't exist
        mock_exists.return_value = False
        mock_input.return_value = "2"  # Manual download choice
        
        with patch('webbrowser.open') as mock_browser:
            result = download_harup_data("/test/data")
            
            assert result is None
            mock_browser.assert_called_once()
    
    @patch('gaitsetpy.dataset.utils.input')
    @patch('gaitsetpy.dataset.utils.os.path.exists')
    def test_download_harup_skip_choice(self, mock_exists, mock_input):
        """Test HAR-UP download with skip choice."""
        # Mock dataset doesn't exist
        mock_exists.return_value = False
        mock_input.return_value = "3"  # Skip download choice
        
        result = download_harup_data("/test/data")
        
        assert result is None
    
    @patch('gaitsetpy.dataset.utils.input')
    @patch('gaitsetpy.dataset.utils.os.path.exists')
    def test_download_harup_invalid_choice(self, mock_exists, mock_input):
        """Test HAR-UP download with invalid choice."""
        # Mock dataset doesn't exist
        mock_exists.return_value = False
        mock_input.return_value = "invalid"  # Invalid choice
        
        result = download_harup_data("/test/data")
        
        assert result is None
    
    @patch('gaitsetpy.dataset.utils.os.path.exists')
    def test_download_harup_dataset_exists(self, mock_exists):
        """Test HAR-UP download when dataset already exists."""
        # Mock dataset exists
        mock_exists.return_value = True
        
        result = download_harup_data("/test/data")
        
        assert result.endswith("DataSet")


class TestExtractDataset:
    """Test cases for the main extract_dataset function."""
    
    def test_extract_daphnet_dataset(self):
        """Test extracting Daphnet dataset."""
        with patch('gaitsetpy.dataset.utils.extract_daphnet_data') as mock_extract:
            extract_dataset("daphnet", "/test/data")
            mock_extract.assert_called_once_with("/test/data")
    
    def test_extract_mobifall_dataset(self):
        """Test extracting MobiFall dataset."""
        with patch('gaitsetpy.dataset.utils.extract_mobifall_data') as mock_extract:
            extract_dataset("mobifall", "/test/data")
            mock_extract.assert_called_once_with("/test/data")
    
    def test_extract_arduous_dataset(self):
        """Test extracting Arduous dataset."""
        with patch('gaitsetpy.dataset.utils.extract_arduous_data') as mock_extract:
            extract_dataset("arduous", "/test/data")
            mock_extract.assert_called_once_with("/test/data")
    
    def test_extract_harup_dataset(self):
        """Test extracting HAR-UP dataset."""
        with patch('gaitsetpy.dataset.utils.extract_harup_data') as mock_extract:
            extract_dataset("harup", "/test/data")
            mock_extract.assert_called_once_with("/test/data")
    
    def test_extract_urfall_dataset(self):
        """Test extracting UrFall dataset."""
        with patch('gaitsetpy.dataset.utils.extract_urfall_data') as mock_extract:
            extract_dataset("urfall", "/test/data")
            mock_extract.assert_called_once_with("/test/data")
    
    def test_extract_physionet_dataset(self):
        """Test extracting PhysioNet dataset (should pass silently)."""
        # PhysioNet is handled by the loader itself, so this should not raise an error
        extract_dataset("physionet", "/test/data")
    
    def test_extract_unsupported_dataset(self):
        """Test extracting unsupported dataset."""
        with pytest.raises(ValueError, match="Dataset unsupported not supported"):
            extract_dataset("unsupported", "/test/data")


class TestExtractDaphnetData:
    """Test cases for Daphnet dataset extraction."""
    
    @patch('gaitsetpy.dataset.utils.zipfile.ZipFile')
    @patch('gaitsetpy.dataset.utils.os.path.join')
    def test_extract_daphnet_success(self, mock_join, mock_zip):
        """Test successful Daphnet extraction."""
        mock_join.return_value = "/test/data/daphnet.zip"
        
        mock_zip_instance = Mock()
        mock_zip.return_value.__enter__.return_value = mock_zip_instance
        mock_zip.return_value.__exit__.return_value = None
        
        extract_daphnet_data("/test/data")
        
        mock_zip.assert_called_once_with("/test/data/daphnet.zip", "r")
        mock_zip_instance.extractall.assert_called_once_with("/test/data")


class TestExtractUrfallData:
    """Test cases for UrFall dataset extraction."""
    
    @patch('gaitsetpy.dataset.utils.zipfile.ZipFile')
    @patch('gaitsetpy.dataset.utils.os.path.exists')
    @patch('gaitsetpy.dataset.utils.os.path.join')
    def test_extract_urfall_success(self, mock_join, mock_exists, mock_zip):
        """Test successful UrFall extraction."""
        # Mock zip file exists
        mock_exists.return_value = True
        mock_join.side_effect = lambda *args: "/".join(args)
        
        mock_zip_instance = Mock()
        mock_zip.return_value.__enter__.return_value = mock_zip_instance
        mock_zip.return_value.__exit__.return_value = None
        
        extract_urfall_data("/test/data", use_falls=True, use_adls=True)
        
        # Should attempt to extract zip files
        assert mock_zip.called
    
    @patch('gaitsetpy.dataset.utils.os.path.exists')
    def test_extract_urfall_no_files(self, mock_exists):
        """Test UrFall extraction when no files exist."""
        # Mock no files exist
        mock_exists.return_value = False
        
        extract_urfall_data("/test/data", use_falls=True, use_adls=True)
        
        # Should not raise an error


class TestExtractHarupData:
    """Test cases for HAR-UP dataset extraction."""
    
    @patch('gaitsetpy.dataset.utils.os.path.exists')
    @patch('gaitsetpy.dataset.utils.zipfile.ZipFile')
    def test_extract_harup_success(self, mock_zip, mock_exists):
        """Test successful HAR-UP extraction."""
        # Mock dataset doesn't exist but zip does
        mock_exists.side_effect = lambda path: path.endswith("DataSet") == False
        
        mock_zip_instance = Mock()
        mock_zip.return_value.__enter__.return_value = mock_zip_instance
        mock_zip.return_value.__exit__.return_value = None
        
        extract_harup_data("/test/data")
        
        mock_zip.assert_called_once()
        mock_zip_instance.extractall.assert_called_once_with("/test/data")
    
    @patch('gaitsetpy.dataset.utils.os.path.exists')
    def test_extract_harup_already_extracted(self, mock_exists):
        """Test HAR-UP extraction when already extracted."""
        # Mock dataset already exists
        mock_exists.return_value = True
        
        extract_harup_data("/test/data")
        
        # Should not attempt extraction
    
    @patch('gaitsetpy.dataset.utils.os.path.exists')
    def test_extract_harup_no_zip(self, mock_exists):
        """Test HAR-UP extraction when zip file doesn't exist."""
        # Mock neither dataset nor zip exist
        mock_exists.return_value = False
        
        extract_harup_data("/test/data")
        
        # Should not raise an error


class TestSlidingWindow:
    """Test cases for the sliding_window utility function."""
    
    def test_sliding_window_basic(self):
        """Test basic sliding window functionality."""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        windows = sliding_window(data, window_size=3, step_size=2)
        
        expected = [
            [1, 2, 3],
            [3, 4, 5],
            [5, 6, 7],
            [7, 8, 9]
        ]
        
        assert windows == expected
    
    def test_sliding_window_step_size_one(self):
        """Test sliding window with step size of 1."""
        data = [1, 2, 3, 4, 5]
        windows = sliding_window(data, window_size=3, step_size=1)
        
        expected = [
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5]
        ]
        
        assert windows == expected
    
    def test_sliding_window_large_step(self):
        """Test sliding window with large step size."""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        windows = sliding_window(data, window_size=3, step_size=5)
        
        expected = [
            [1, 2, 3],
            [6, 7, 8]
        ]
        
        assert windows == expected
    
    def test_sliding_window_exact_fit(self):
        """Test sliding window when data fits exactly."""
        data = [1, 2, 3, 4, 5, 6]
        windows = sliding_window(data, window_size=3, step_size=3)
        
        expected = [
            [1, 2, 3],
            [4, 5, 6]
        ]
        
        assert windows == expected
    
    def test_sliding_window_single_window(self):
        """Test sliding window with single window."""
        data = [1, 2, 3]
        windows = sliding_window(data, window_size=3, step_size=1)
        
        expected = [[1, 2, 3]]
        
        assert windows == expected
    
    def test_sliding_window_empty_data(self):
        """Test sliding window with empty data."""
        data = []
        windows = sliding_window(data, window_size=3, step_size=1)
        
        assert windows == []
    
    def test_sliding_window_insufficient_data(self):
        """Test sliding window with insufficient data."""
        data = [1, 2]
        windows = sliding_window(data, window_size=3, step_size=1)
        
        assert windows == []
    
    def test_sliding_window_numpy_array(self):
        """Test sliding window with numpy array."""
        import numpy as np
        
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        windows = sliding_window(data, window_size=3, step_size=2)
        
        expected = [
            [1, 2, 3],
            [3, 4, 5],
            [5, 6, 7],
            [7, 8, 9]
        ]
        
        # Convert numpy arrays to lists for comparison
        windows_lists = [list(w) for w in windows]
        assert windows_lists == expected


class TestDownloadFile:
    """Test cases for the _download_file utility function."""
    
    @patch('gaitsetpy.dataset.utils.requests.get')
    @patch('tqdm.tqdm')
    @patch('builtins.open', new_callable=mock_open)
    @patch('gaitsetpy.dataset.utils.os.path.getsize')
    def test_download_file_success(self, mock_getsize, mock_file, mock_tqdm, mock_get):
        """Test successful file download."""
        # Mock HTTP response
        mock_response = Mock()
        mock_response.headers = {'content-length': '1024'}
        mock_response.iter_content.return_value = [b'chunk1', b'chunk2']
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Mock file size
        mock_getsize.return_value = 1024
        
        # Mock tqdm
        mock_progress = Mock()
        mock_tqdm.return_value = mock_progress
        
        success, result = _download_file("http://example.com/file.txt", "/test/file.txt", "file.txt")
        
        assert success is True
        assert result == "/test/file.txt"
        mock_get.assert_called_once_with("http://example.com/file.txt", stream=True, timeout=60)
    
    @patch('gaitsetpy.dataset.utils.requests.get')
    @patch('builtins.open', new_callable=mock_open)
    @patch('gaitsetpy.dataset.utils.os.path.exists')
    def test_download_file_http_error(self, mock_exists, mock_file, mock_get):
        """Test file download with HTTP error."""
        # Mock HTTP error
        mock_get.side_effect = requests.exceptions.HTTPError("404 Not Found")
        
        # Mock file doesn't exist initially
        mock_exists.return_value = False
        
        success, result = _download_file("http://example.com/file.txt", "/test/file.txt", "file.txt")
        
        assert success is False
        assert "404 Not Found" in result
    
    @patch('gaitsetpy.dataset.utils.requests.get')
    @patch('builtins.open', new_callable=mock_open)
    @patch('gaitsetpy.dataset.utils.os.path.exists')
    def test_download_file_timeout(self, mock_exists, mock_file, mock_get):
        """Test file download with timeout."""
        # Mock timeout error
        mock_get.side_effect = requests.exceptions.Timeout("Request timed out")
        
        # Mock file doesn't exist initially
        mock_exists.return_value = False
        
        success, result = _download_file("http://example.com/file.txt", "/test/file.txt", "file.txt")
        
        assert success is False
        assert "Request timed out" in result
    
    @patch('gaitsetpy.dataset.utils.requests.get')
    @patch('builtins.open', new_callable=mock_open)
    @patch('gaitsetpy.dataset.utils.os.path.exists')
    def test_download_file_io_error(self, mock_exists, mock_file, mock_get):
        """Test file download with IO error."""
        # Mock HTTP response
        mock_response = Mock()
        mock_response.headers = {'content-length': '1024'}
        mock_response.iter_content.return_value = [b'chunk1', b'chunk2']
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Mock IO error
        mock_file.side_effect = IOError("Permission denied")
        
        # Mock file doesn't exist initially
        mock_exists.return_value = False
        
        success, result = _download_file("http://example.com/file.txt", "/test/file.txt", "file.txt")
        
        assert success is False
        assert "Permission denied" in result
    
    @patch('gaitsetpy.dataset.utils.requests.get')
    @patch('tqdm.tqdm')
    @patch('builtins.open', new_callable=mock_open)
    @patch('gaitsetpy.dataset.utils.os.path.getsize')
    def test_download_file_incomplete(self, mock_getsize, mock_file, mock_tqdm, mock_get):
        """Test file download with incomplete file."""
        # Mock HTTP response
        mock_response = Mock()
        mock_response.headers = {'content-length': '1024'}
        mock_response.iter_content.return_value = [b'chunk1', b'chunk2']
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Mock file size smaller than expected
        mock_getsize.return_value = 512  # Less than 1024
        
        # Mock tqdm
        mock_progress = Mock()
        mock_tqdm.return_value = mock_progress
        
        success, result = _download_file("http://example.com/file.txt", "/test/file.txt", "file.txt")
        
        assert success is False
        assert "Incomplete download" in result
    
    @patch('gaitsetpy.dataset.utils.requests.get')
    @patch('builtins.open', new_callable=mock_open)
    @patch('gaitsetpy.dataset.utils.os.path.exists')
    @patch('gaitsetpy.dataset.utils.os.remove')
    def test_download_file_cleanup_on_error(self, mock_remove, mock_exists, mock_file, mock_get):
        """Test that partial files are cleaned up on error."""
        # Mock HTTP error
        mock_get.side_effect = requests.exceptions.RequestException("Network error")
        
        # Mock file doesn't exist initially
        mock_exists.return_value = False
        
        success, result = _download_file("http://example.com/file.txt", "/test/file.txt", "file.txt")
        
        assert success is False
        # Should attempt to remove the partial file
        mock_remove.assert_called_once_with("/test/file.txt")


class TestDatasetUtilsEdgeCases:
    """Test edge cases for dataset utility functions."""
    
    def test_download_mobifall_data_placeholder(self):
        """Test that MobiFall download is a placeholder."""
        # This should not raise an error but also not do anything
        download_mobifall_data("/test/data")
    
    def test_download_arduous_data_placeholder(self):
        """Test that Arduous download is a placeholder."""
        # This should not raise an error but also not do anything
        download_arduous_data("/test/data")
    
    def test_extract_mobifall_data_placeholder(self):
        """Test that MobiFall extraction is a placeholder."""
        # This should not raise an error but also not do anything
        extract_mobifall_data("/test/data")
    
    def test_extract_arduous_data_placeholder(self):
        """Test that Arduous extraction is a placeholder."""
        # This should not raise an error but also not do anything
        extract_arduous_data("/test/data")
    
    def test_sliding_window_zero_window_size(self):
        """Test sliding window with zero window size."""
        data = [1, 2, 3, 4, 5]
        windows = sliding_window(data, window_size=0, step_size=1)
        
        assert windows == []
    
    def test_sliding_window_zero_step_size(self):
        """Test sliding window with zero step size."""
        data = [1, 2, 3, 4, 5]
        windows = sliding_window(data, window_size=3, step_size=0)
        
        # Zero step size should return empty list due to division by zero protection
        assert windows == []
    
    def test_sliding_window_negative_parameters(self):
        """Test sliding window with negative parameters."""
        data = [1, 2, 3, 4, 5]
        
        # Negative window size should result in empty list
        windows = sliding_window(data, window_size=-1, step_size=1)
        assert windows == []
        
        # Negative step size should result in empty list
        windows = sliding_window(data, window_size=3, step_size=-1)
        assert windows == []
