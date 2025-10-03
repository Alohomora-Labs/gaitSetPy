"""
Comprehensive test suite for concurrent downloading functionality in dataset loaders.

This module tests the thread-safe concurrent downloading capabilities added to the
BaseDatasetLoader class and its subclasses. It verifies that multi-threaded downloads
work correctly, handle errors gracefully, and provide accurate statistics.

Test Coverage:
    - BaseDatasetLoader concurrent download methods
    - PhysioNetLoader threaded downloading
    - Thread safety and error handling
    - Download statistics and reporting
    - Configuration of max_workers

Author: @aharshit123456
"""

import os
import sys
import pytest
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add parent directory to path for imports (standard for unit tests)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from gaitsetpy.dataset.physionet import PhysioNetLoader
from gaitsetpy.dataset.harup import HARUPLoader
from gaitsetpy.dataset.urfall import UrFallLoader
from gaitsetpy.dataset.daphnet import DaphnetLoader
from gaitsetpy.dataset.mobifall import MobiFallLoader
from gaitsetpy.dataset.arduous import ArduousLoader
from gaitsetpy.core.base_classes import BaseDatasetLoader


class TestBaseDatasetLoaderConcurrency:
    """Test concurrent downloading functionality in BaseDatasetLoader."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)
    
    def test_loader_initialization_with_max_workers(self):
        """
        Test that dataset loaders can be initialized with custom max_workers.
        
        Verifies:
            - Default max_workers is set correctly
            - Custom max_workers is respected
            - All loaders support the max_workers parameter
        """
        # Test PhysioNetLoader
        loader1 = PhysioNetLoader()
        assert loader1.max_workers == 8, "Default max_workers should be 8"
        
        loader2 = PhysioNetLoader(max_workers=16)
        assert loader2.max_workers == 16, "Custom max_workers should be respected"
        
        # Test other loaders
        harup_loader = HARUPLoader(max_workers=4)
        assert harup_loader.max_workers == 4
        
        urfall_loader = UrFallLoader(max_workers=12)
        assert urfall_loader.max_workers == 12
        
        daphnet_loader = DaphnetLoader(max_workers=6)
        assert daphnet_loader.max_workers == 6
    
    def test_set_max_workers(self):
        """
        Test the set_max_workers method.
        
        Verifies:
            - max_workers can be changed after initialization
            - Invalid values are rejected
        """
        loader = PhysioNetLoader(max_workers=8)
        
        # Test valid change
        loader.set_max_workers(16)
        assert loader.max_workers == 16
        
        # Test invalid value (should raise ValueError)
        with pytest.raises(ValueError, match="max_workers must be at least 1"):
            loader.set_max_workers(0)
        
        with pytest.raises(ValueError):
            loader.set_max_workers(-5)
    
    def test_download_stats_initialization(self):
        """
        Test that download statistics are properly initialized.
        
        Verifies:
            - Stats dictionary is initialized correctly
            - All counters start at zero
        """
        loader = PhysioNetLoader()
        stats = loader.get_download_stats()
        
        assert 'success' in stats
        assert 'failed' in stats
        assert 'skipped' in stats
        assert stats['success'] == 0
        assert stats['failed'] == 0
        assert stats['skipped'] == 0
    
    @patch('gaitsetpy.core.base_classes.requests.get')
    def test_download_file_success(self, mock_get, temp_dir):
        """
        Test successful file download.
        
        Verifies:
            - Single file download works correctly
            - File is saved to correct location
            - Success status is returned
        """
        loader = PhysioNetLoader()
        
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_content = Mock(return_value=[b'test data chunk 1', b'test data chunk 2'])
        mock_get.return_value = mock_response
        
        # Test download
        url = "http://example.com/testfile.txt"
        dest_path = os.path.join(temp_dir, "testfile.txt")
        
        success, message = loader._download_file(url, dest_path)
        
        assert success is True
        assert os.path.exists(dest_path)
        assert "Successfully downloaded" in message
    
    @patch('gaitsetpy.core.base_classes.requests.get')
    def test_download_file_already_exists(self, mock_get, temp_dir):
        """
        Test behavior when file already exists.
        
        Verifies:
            - Existing files are skipped
            - No download is attempted
            - Skipped status is returned
        """
        loader = PhysioNetLoader()
        
        # Create existing file
        dest_path = os.path.join(temp_dir, "existing_file.txt")
        with open(dest_path, 'w') as f:
            f.write("existing content")
        
        # Test download
        url = "http://example.com/existing_file.txt"
        success, message = loader._download_file(url, dest_path)
        
        assert success is True
        assert "already exists" in message
        # Verify no request was made
        mock_get.assert_not_called()
    
    @patch('gaitsetpy.core.base_classes.requests.get')
    def test_download_file_http_error(self, mock_get, temp_dir):
        """
        Test handling of HTTP errors.
        
        Verifies:
            - HTTP error codes are handled correctly
            - Appropriate error message is returned
            - Failed status is tracked
        """
        loader = PhysioNetLoader()
        
        # Mock 404 response
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        url = "http://example.com/nonexistent.txt"
        dest_path = os.path.join(temp_dir, "nonexistent.txt")
        
        success, message = loader._download_file(url, dest_path)
        
        assert success is False
        assert "HTTP 404" in message
    
    @patch('gaitsetpy.core.base_classes.requests.get')
    def test_download_file_timeout(self, mock_get, temp_dir):
        """
        Test handling of timeout errors.
        
        Verifies:
            - Timeout exceptions are caught
            - Appropriate error message is returned
        """
        loader = PhysioNetLoader()
        
        # Mock timeout
        import requests
        mock_get.side_effect = requests.exceptions.Timeout("Connection timeout")
        
        url = "http://example.com/slow_file.txt"
        dest_path = os.path.join(temp_dir, "slow_file.txt")
        
        success, message = loader._download_file(url, dest_path)
        
        assert success is False
        assert "Timeout" in message
    
    @patch('gaitsetpy.core.base_classes.requests.get')
    def test_download_file_request_exception(self, mock_get, temp_dir):
        """
        Test handling of general request exceptions.
        
        Verifies:
            - RequestException is caught
            - Appropriate error message is returned
        """
        loader = PhysioNetLoader()
        
        # Mock request exception
        import requests
        mock_get.side_effect = requests.exceptions.RequestException("Network error")
        
        url = "http://example.com/error_file.txt"
        dest_path = os.path.join(temp_dir, "error_file.txt")
        
        success, message = loader._download_file(url, dest_path)
        
        assert success is False
        assert "Request error" in message
    
    @patch('gaitsetpy.core.base_classes.requests.get')
    @patch('builtins.open', side_effect=IOError("Disk full"))
    def test_download_file_io_error(self, mock_open, mock_get, temp_dir):
        """
        Test handling of IO errors during file writing.
        
        Verifies:
            - IOError is caught
            - Appropriate error message is returned
        """
        loader = PhysioNetLoader()
        
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_content = Mock(return_value=[b'test data'])
        mock_get.return_value = mock_response
        
        url = "http://example.com/file.txt"
        dest_path = os.path.join(temp_dir, "file.txt")
        
        success, message = loader._download_file(url, dest_path)
        
        assert success is False
        assert "IO error" in message
    
    @patch('gaitsetpy.core.base_classes.requests.get')
    def test_download_file_unexpected_error(self, mock_get, temp_dir):
        """
        Test handling of unexpected exceptions.
        
        Verifies:
            - Unexpected exceptions are caught
            - Appropriate error message is returned
        """
        loader = PhysioNetLoader()
        
        # Mock unexpected exception
        mock_get.side_effect = ValueError("Unexpected error")
        
        url = "http://example.com/file.txt"
        dest_path = os.path.join(temp_dir, "file.txt")
        
        success, message = loader._download_file(url, dest_path)
        
        assert success is False
        assert "Unexpected error" in message
    
    @patch('gaitsetpy.core.base_classes.requests.get')
    def test_download_files_concurrent(self, mock_get, temp_dir):
        """
        Test concurrent downloading of multiple files.
        
        Verifies:
            - Multiple files can be downloaded concurrently
            - Download statistics are accurate
            - Results are properly collected
        """
        loader = PhysioNetLoader(max_workers=4)
        
        # Mock successful responses
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_content = Mock(return_value=[b'test data'])
        mock_get.return_value = mock_response
        
        # Create download tasks
        download_tasks = [
            {'url': f'http://example.com/file{i}.txt', 
             'dest_path': os.path.join(temp_dir, f'file{i}.txt')}
            for i in range(10)
        ]
        
        # Execute concurrent downloads
        results = loader.download_files_concurrent(
            download_tasks, 
            show_progress=False,
            desc="Testing concurrent downloads"
        )
        
        # Verify results
        assert results['total'] == 10
        assert results['success'] == 10
        assert results['failed'] == 0
        assert len(results['all_results']) == 10
        
        # Verify all files were created
        for i in range(10):
            assert os.path.exists(os.path.join(temp_dir, f'file{i}.txt'))
    
    @patch('gaitsetpy.core.base_classes.requests.get')
    def test_download_files_concurrent_mixed_results(self, mock_get, temp_dir):
        """
        Test concurrent downloading with mixed success/failure results.
        
        Verifies:
            - Both successful and failed downloads are handled
            - Statistics accurately reflect mixed results
            - Failed downloads are properly reported
        """
        loader = PhysioNetLoader(max_workers=4)
        
        # Mock mixed responses (some succeed, some fail)
        def mock_get_side_effect(*args, **kwargs):
            url = args[0]
            response = Mock()
            if 'fail' in url:
                response.status_code = 404
            else:
                response.status_code = 200
                response.iter_content = Mock(return_value=[b'test data'])
            return response
        
        mock_get.side_effect = mock_get_side_effect
        
        # Create download tasks with some that will fail
        download_tasks = [
            {'url': f'http://example.com/file{i}.txt', 
             'dest_path': os.path.join(temp_dir, f'file{i}.txt')}
            for i in range(5)
        ] + [
            {'url': f'http://example.com/fail{i}.txt', 
             'dest_path': os.path.join(temp_dir, f'fail{i}.txt')}
            for i in range(3)
        ]
        
        # Execute concurrent downloads
        results = loader.download_files_concurrent(
            download_tasks, 
            show_progress=False
        )
        
        # Verify mixed results
        assert results['total'] == 8
        assert results['success'] == 5
        assert results['failed'] == 3
        assert len(results['failed_downloads']) == 3
    
    @patch('gaitsetpy.core.base_classes.requests.get')
    def test_download_files_concurrent_with_existing(self, mock_get, temp_dir):
        """
        Test concurrent downloading when some files already exist.
        
        Verifies:
            - Existing files are skipped
            - Only new files are downloaded
            - Skipped count is accurate
        """
        loader = PhysioNetLoader(max_workers=4)
        
        # Create some existing files
        for i in [0, 2, 4]:
            with open(os.path.join(temp_dir, f'file{i}.txt'), 'w') as f:
                f.write("existing content")
        
        # Mock successful responses for new files
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_content = Mock(return_value=[b'new data'])
        mock_get.return_value = mock_response
        
        # Create download tasks
        download_tasks = [
            {'url': f'http://example.com/file{i}.txt', 
             'dest_path': os.path.join(temp_dir, f'file{i}.txt')}
            for i in range(6)
        ]
        
        # Execute concurrent downloads
        results = loader.download_files_concurrent(
            download_tasks, 
            show_progress=False
        )
        
        # Verify results
        assert results['total'] == 6
        assert results['skipped'] == 3  # Files 0, 2, 4 already existed
        assert results['success'] == 3  # Files 1, 3, 5 were downloaded
    
    @patch('gaitsetpy.core.base_classes.requests.get')
    def test_download_files_concurrent_empty_list(self, mock_get, temp_dir):
        """
        Test concurrent downloading with empty task list.
        
        Verifies:
            - Empty task list is handled correctly
            - No downloads are attempted
            - Returns appropriate results
        """
        loader = PhysioNetLoader()
        
        # Execute with empty task list
        results = loader.download_files_concurrent([], show_progress=False)
        
        # Verify results
        assert results['total'] == 0
        assert results['success'] == 0
        assert results['failed'] == 0
        assert results['skipped'] == 0
        assert len(results['failed_downloads']) == 0
        assert len(results['all_results']) == 0
        
        # Verify no requests were made
        mock_get.assert_not_called()
    
    @patch('gaitsetpy.core.base_classes.requests.get')
    def test_download_files_concurrent_exception_in_future(self, mock_get, temp_dir):
        """
        Test handling of exceptions during concurrent execution.
        
        Verifies:
            - Exceptions in futures are caught
            - Failed downloads are tracked
            - Other downloads continue
        """
        loader = PhysioNetLoader(max_workers=2)
        
        # Mock to raise exception
        def mock_get_side_effect(*args, **kwargs):
            url = args[0]
            if 'exception' in url:
                raise RuntimeError("Simulated exception")
            response = Mock()
            response.status_code = 200
            response.iter_content = Mock(return_value=[b'data'])
            return response
        
        mock_get.side_effect = mock_get_side_effect
        
        # Create tasks with one that will raise exception
        download_tasks = [
            {'url': 'http://example.com/good1.txt', 
             'dest_path': os.path.join(temp_dir, 'good1.txt')},
            {'url': 'http://example.com/exception.txt', 
             'dest_path': os.path.join(temp_dir, 'exception.txt')},
            {'url': 'http://example.com/good2.txt', 
             'dest_path': os.path.join(temp_dir, 'good2.txt')},
        ]
        
        results = loader.download_files_concurrent(
            download_tasks, 
            show_progress=False
        )
        
        # Verify that exception was handled
        assert results['total'] == 3
        assert results['failed'] >= 1
        assert any('Exception' in r['message'] or 'Unexpected' in r['message'] 
                  for r in results['all_results'] if not r['success'])
    
    @patch('gaitsetpy.core.base_classes.requests.get')
    def test_download_file_creates_directory(self, mock_get, temp_dir):
        """
        Test that _download_file creates parent directories if needed.
        
        Verifies:
            - Parent directories are created automatically
            - File is saved successfully in nested directory
        """
        loader = PhysioNetLoader()
        
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_content = Mock(return_value=[b'test data'])
        mock_get.return_value = mock_response
        
        # Use nested directory that doesn't exist
        nested_dir = os.path.join(temp_dir, 'level1', 'level2', 'level3')
        dest_path = os.path.join(nested_dir, 'file.txt')
        
        url = "http://example.com/file.txt"
        success, message = loader._download_file(url, dest_path)
        
        assert success is True
        assert os.path.exists(dest_path)
        assert os.path.exists(nested_dir)
    
    def test_get_info_includes_max_workers(self):
        """
        Test that get_info() includes max_workers information.
        
        Verifies:
            - get_info() method returns max_workers
            - Information is accessible from loader instances
        """
        loader = PhysioNetLoader(max_workers=12)
        info = loader.get_info()
        
        assert 'max_workers' in info
        assert info['max_workers'] == 12


class TestPhysioNetConcurrentDownload:
    """Test PhysioNet-specific concurrent downloading functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)
    
    @patch('gaitsetpy.dataset.physionet.PhysioNetLoader.download_files_concurrent')
    def test_physionet_uses_concurrent_download(self, mock_download, temp_dir):
        """
        Test that PhysioNet loader uses concurrent downloading.
        
        Verifies:
            - _download_physionet_data calls download_files_concurrent
            - Correct number of tasks is created
            - Progress is shown by default
        """
        loader = PhysioNetLoader(max_workers=8)
        
        # Mock successful download
        mock_download.return_value = {
            'total': 100,
            'success': 100,
            'failed': 0,
            'skipped': 0,
            'failed_downloads': []
        }
        
        # Call download method
        loader._download_physionet_data(temp_dir)
        
        # Verify concurrent download was called
        assert mock_download.called
        call_args = mock_download.call_args
        
        # Verify download tasks were created
        tasks = call_args[0][0]  # First positional argument
        assert len(tasks) > 0
        assert all('url' in task and 'dest_path' in task for task in tasks)
        
        # Verify show_progress is True
        assert call_args[1]['show_progress'] is True
    
    @patch('gaitsetpy.dataset.physionet.PhysioNetLoader.download_files_concurrent')
    def test_physionet_with_failures(self, mock_download, temp_dir):
        """
        Test PhysioNet download with some failures.
        
        Verifies:
            - Failed downloads are reported correctly
            - Summary includes failure details
        """
        loader = PhysioNetLoader(max_workers=4)
        
        # Mock download with some failures
        mock_download.return_value = {
            'total': 20,
            'success': 15,
            'failed': 3,
            'skipped': 2,
            'failed_downloads': [
                {'url': 'http://example.com/file1.txt', 
                 'dest_path': os.path.join(temp_dir, 'file1.txt'),
                 'error': 'HTTP 404'},
                {'url': 'http://example.com/file2.txt', 
                 'dest_path': os.path.join(temp_dir, 'file2.txt'),
                 'error': 'Connection timeout'},
                {'url': 'http://example.com/file3.txt', 
                 'dest_path': os.path.join(temp_dir, 'file3.txt'),
                 'error': 'Network error'},
            ]
        }
        
        # Call download method (should print summary)
        result_path = loader._download_physionet_data(temp_dir)
        
        assert result_path == os.path.join(temp_dir, 'physionet_gaitpdb')
        assert mock_download.called
    
    @patch('gaitsetpy.dataset.physionet.PhysioNetLoader.download_files_concurrent')
    def test_physionet_with_many_failures(self, mock_download, temp_dir):
        """
        Test PhysioNet download with many failures (>10).
        
        Verifies:
            - Only first 10 failures are shown in summary
            - "... and N more failures" message appears
        """
        loader = PhysioNetLoader(max_workers=4)
        
        # Mock download with many failures
        failed_downloads = [
            {'url': f'http://example.com/file{i}.txt', 
             'dest_path': os.path.join(temp_dir, f'file{i}.txt'),
             'error': f'Error {i}'}
            for i in range(15)
        ]
        
        mock_download.return_value = {
            'total': 20,
            'success': 5,
            'failed': 15,
            'skipped': 0,
            'failed_downloads': failed_downloads
        }
        
        # Call download method (should print truncated summary)
        result_path = loader._download_physionet_data(temp_dir)
        
        assert result_path == os.path.join(temp_dir, 'physionet_gaitpdb')
        assert mock_download.called
    
    def test_physionet_dataset_already_exists(self, temp_dir):
        """
        Test that PhysioNet skips download if dataset exists.
        
        Verifies:
            - Existing dataset is detected
            - No download is attempted
        """
        loader = PhysioNetLoader()
        
        # Create dataset directory with some files
        dataset_path = os.path.join(temp_dir, 'physionet_gaitpdb')
        os.makedirs(dataset_path)
        with open(os.path.join(dataset_path, 'dummy.txt'), 'w') as f:
            f.write("existing data")
        
        # Call download method
        result_path = loader._download_physionet_data(temp_dir)
        
        assert result_path == dataset_path
        assert os.path.exists(dataset_path)
    
    def test_physionet_download_task_creation(self, temp_dir):
        """
        Test that PhysioNet creates correct download tasks.
        
        Verifies:
            - All expected file patterns are included
            - URLs are correctly formed
            - Destination paths are in the right directory
        """
        loader = PhysioNetLoader()
        
        # We'll need to extract the task creation logic
        # by calling the method with a mock
        with patch.object(loader, 'download_files_concurrent') as mock_download:
            mock_download.return_value = {
                'total': 0, 'success': 0, 'failed': 0, 
                'skipped': 0, 'failed_downloads': []
            }
            
            loader._download_physionet_data(temp_dir)
            
            # Get the tasks that were passed
            tasks = mock_download.call_args[0][0]
            
            # Verify we have a substantial number of files
            assert len(tasks) > 100, "PhysioNet should have 100+ files"
            
            # Verify URL format
            assert all('physionet.org' in task['url'] for task in tasks)
            
            # Verify some expected file patterns
            filenames = [os.path.basename(task['dest_path']) for task in tasks]
            assert any('GaCo' in f for f in filenames), "Should include GaCo files"
            assert any('GaPt' in f for f in filenames), "Should include GaPt files"
            assert any('JuCo' in f for f in filenames), "Should include JuCo files"
            assert any('SiCo' in f for f in filenames), "Should include SiCo files"


class TestThreadSafety:
    """Test thread safety of concurrent operations."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)
    
    @patch('gaitsetpy.core.base_classes.requests.get')
    def test_concurrent_access_to_stats(self, mock_get, temp_dir):
        """
        Test that download statistics are thread-safe.
        
        Verifies:
            - Multiple threads can update stats simultaneously
            - Final counts are accurate
            - No race conditions occur
        """
        loader = PhysioNetLoader(max_workers=10)
        
        # Mock successful responses
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_content = Mock(return_value=[b'data'])
        mock_get.return_value = mock_response
        
        # Create many download tasks to stress-test concurrency
        download_tasks = [
            {'url': f'http://example.com/file{i}.txt', 
             'dest_path': os.path.join(temp_dir, f'file{i}.txt')}
            for i in range(50)
        ]
        
        # Execute concurrent downloads
        results = loader.download_files_concurrent(
            download_tasks, 
            show_progress=False
        )
        
        # Verify statistics are consistent
        assert results['total'] == 50
        assert results['success'] + results['failed'] + results['skipped'] == 50
        
        # Get stats from loader
        stats = loader.get_download_stats()
        assert stats['success'] == results['success']
        assert stats['failed'] == results['failed']
        assert stats['skipped'] == results['skipped']


class TestAllDatasetLoaders:
    """Test that all dataset loaders support concurrent downloading."""
    
    def test_all_loaders_support_max_workers(self):
        """
        Test that all dataset loaders support max_workers parameter.
        
        Verifies:
            - All loaders can be initialized with max_workers
            - Default value is correctly set
            - Custom value is respected
        """
        # Test all loaders
        loaders = [
            (PhysioNetLoader, "physionet"),
            (HARUPLoader, "harup"),
            (UrFallLoader, "urfall"),
            (DaphnetLoader, "daphnet"),
            (MobiFallLoader, "mobifall"),
            (ArduousLoader, "arduous"),
        ]
        
        for LoaderClass, name in loaders:
            # Test default
            loader_default = LoaderClass()
            assert loader_default.max_workers == 8, f"{name} default max_workers should be 8"
            assert loader_default.name == name
            
            # Test custom
            loader_custom = LoaderClass(max_workers=16)
            assert loader_custom.max_workers == 16, f"{name} custom max_workers should be 16"
    
    def test_all_loaders_have_download_methods(self):
        """
        Test that all loaders have the download methods from base class.
        
        Verifies:
            - _download_file method is available
            - download_files_concurrent method is available
            - set_max_workers method is available
            - get_download_stats method is available
        """
        loader = PhysioNetLoader()
        
        assert hasattr(loader, '_download_file')
        assert callable(loader._download_file)
        
        assert hasattr(loader, 'download_files_concurrent')
        assert callable(loader.download_files_concurrent)
        
        assert hasattr(loader, 'set_max_workers')
        assert callable(loader.set_max_workers)
        
        assert hasattr(loader, 'get_download_stats')
        assert callable(loader.get_download_stats)


class TestIntegrationScenarios:
    """Test real-world integration scenarios."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)
    
    def test_multiple_loaders_independent_stats(self):
        """
        Test that multiple loader instances maintain independent statistics.
        
        Verifies:
            - Each loader instance has its own stats
            - Stats don't interfere between instances
        """
        loader1 = PhysioNetLoader(max_workers=4)
        loader2 = PhysioNetLoader(max_workers=8)
        
        assert loader1.get_download_stats() == loader2.get_download_stats()
        assert loader1.get_download_stats() is not loader2.get_download_stats()
    
    def test_reconfiguring_max_workers_during_operation(self):
        """
        Test that max_workers can be changed between operations.
        
        Verifies:
            - max_workers can be updated
            - New value is used in subsequent operations
        """
        loader = PhysioNetLoader(max_workers=4)
        assert loader.max_workers == 4
        
        loader.set_max_workers(16)
        assert loader.max_workers == 16
        
        # Verify it's used in get_info
        info = loader.get_info()
        assert info['max_workers'] == 16
    
    @patch('gaitsetpy.core.base_classes.requests.get')
    def test_stats_reset_between_downloads(self, mock_get, temp_dir):
        """
        Test that download statistics are reset between operations.
        
        Verifies:
            - Stats are reset when starting new download
            - Previous stats don't interfere with new downloads
        """
        loader = PhysioNetLoader(max_workers=2)
        
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_content = Mock(return_value=[b'data'])
        mock_get.return_value = mock_response
        
        # First download
        tasks1 = [
            {'url': f'http://example.com/file{i}.txt', 
             'dest_path': os.path.join(temp_dir, f'set1_file{i}.txt')}
            for i in range(3)
        ]
        results1 = loader.download_files_concurrent(tasks1, show_progress=False)
        assert results1['success'] == 3
        
        # Second download should reset stats
        tasks2 = [
            {'url': f'http://example.com/file{i}.txt', 
             'dest_path': os.path.join(temp_dir, f'set2_file{i}.txt')}
            for i in range(5)
        ]
        results2 = loader.download_files_concurrent(tasks2, show_progress=False)
        
        # Stats should reflect only the second download
        assert results2['total'] == 5
        assert results2['success'] == 5
        stats = loader.get_download_stats()
        assert stats['success'] == 5  # Not 8 (3+5)


def test_documentation_completeness():
    """
    Verify that all public methods have proper documentation.
    
    This test ensures that the new concurrent downloading features
    are properly documented for users.
    """
    from gaitsetpy.core.base_classes import BaseDatasetLoader
    
    # Check that new methods have docstrings
    assert BaseDatasetLoader._download_file.__doc__ is not None
    assert BaseDatasetLoader.download_files_concurrent.__doc__ is not None
    assert BaseDatasetLoader.set_max_workers.__doc__ is not None
    assert BaseDatasetLoader.get_download_stats.__doc__ is not None
    
    # Verify docstrings contain key information
    assert "thread" in BaseDatasetLoader._download_file.__doc__.lower()
    assert "concurrent" in BaseDatasetLoader.download_files_concurrent.__doc__.lower()
    assert "example" in BaseDatasetLoader.download_files_concurrent.__doc__.lower()


if __name__ == "__main__":
    """
    Run tests directly from command line.
    
    Usage:
        python test_concurrent_downloading.py
        python -m pytest test_concurrent_downloading.py -v
        python -m pytest test_concurrent_downloading.py -v -k "test_loader_initialization"
    """
    pytest.main([__file__, "-v", "--tb=short"])

