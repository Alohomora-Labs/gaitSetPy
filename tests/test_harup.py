# import pytest
# import pandas as pd
# import numpy as np
# from unittest.mock import patch, MagicMock
# from gaitsetpy.dataset.harup import HARUPLoader

# def test_haruploader_instantiation():
#     loader = HARUPLoader()
#     assert loader.name == "harup"
#     assert "sensors" in loader.metadata

# def test_get_supported_formats():
#     loader = HARUPLoader()
#     assert loader.get_supported_formats() == ['.csv']

# def test_get_sensor_info():
#     loader = HARUPLoader()
#     info = loader.get_sensor_info()
#     assert 'sensors' in info
#     assert 'components' in info
#     assert 'sampling_frequency' in info

# def test_get_activity_info():
#     loader = HARUPLoader()
#     act = loader.get_activity_info()
#     assert 1 in act and 11 in act

# def test_create_sliding_windows_with_dummy_data():
#     loader = HARUPLoader()
#     # Create dummy DataFrame
#     df = pd.DataFrame({
#         'Belt_Acc_X': np.arange(20),
#         'Belt_Acc_Y': np.arange(20),
#         'Belt_Acc_Z': np.arange(20),
#         'activity_id': np.ones(20),
#         'subject_id': np.ones(20),
#         'trial_id': np.ones(20),
#         'activity_label': ['Walking']*20,
#         'TIME': np.arange(20)
#     })
#     data = [df]
#     names = ["dummy"]
#     windows = loader.create_sliding_windows(data, names, window_size=5, step_size=2)
#     assert isinstance(windows, list)
#     assert windows[0]["name"] == "dummy"
#     assert any(w["name"] == "activity_id" for w in windows[0]["windows"])

# @patch("gaitsetpy.dataset.harup.download_dataset")
# @patch("gaitsetpy.dataset.harup.extract_dataset")
# @patch("os.path.exists")
# @patch("os.listdir")
# def test_load_data_mocks(listdir_mock, exists_mock, extract_mock, download_mock):
#     loader = HARUPLoader()
#     exists_mock.return_value = True
#     listdir_mock.return_value = ["UP_Fall_Detection_Dataset"]
#     # Patch os.path.isdir to always return True
#     with patch("os.path.isdir", return_value=True):
#         # Patch pandas.read_csv to return a dummy DataFrame
#         with patch("pandas.read_csv") as read_csv_mock:
#             df = pd.DataFrame({
#                 'Belt_Acc_X': np.arange(10),
#                 'Belt_Acc_Y': np.arange(10),
#                 'Belt_Acc_Z': np.arange(10),
#                 'activity_id': np.ones(10),
#                 'subject_id': np.ones(10),
#                 'trial_id': np.ones(10),
#                 'activity_label': ['Walking']*10,
#                 'TIME': np.arange(10)
#             })
#             read_csv_mock.return_value = df
#             data, names = loader.load_data("/tmp", subjects=[1], activities=[1], trials=[1])
#             assert isinstance(data, list)
#             assert isinstance(names, list) 