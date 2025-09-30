# import pytest
# import pandas as pd
# import numpy as np
# from unittest.mock import patch, MagicMock
# from gaitsetpy.dataset.physionet import PhysioNetLoader

# def test_physionetloader_instantiation():
#     loader = PhysioNetLoader()
#     assert loader.name == "physionet"
#     assert "sensors" in loader.metadata

# def test_get_supported_formats():
#     loader = PhysioNetLoader()
#     assert loader.get_supported_formats() == ['.txt']

# def test_get_sensor_info():
#     loader = PhysioNetLoader()
#     info = loader.get_sensor_info()
#     assert 'sensors' in info
#     assert 'sampling_frequency' in info
#     assert 'window_size' in info

# def test_get_subject_info():
#     loader = PhysioNetLoader()
#     subj = loader.get_subject_info()
#     assert 'Co' in subj and 'Pt' in subj

# def test_create_sliding_windows_with_dummy_data():
#     loader = PhysioNetLoader()
#     # Create dummy DataFrame
#     df = pd.DataFrame({
#         'VGRF_L1': np.arange(20),
#         'VGRF_L2': np.arange(20),
#         'VGRF_L3': np.arange(20),
#         'VGRF_L4': np.arange(20),
#         'VGRF_L5': np.arange(20),
#         'VGRF_L6': np.arange(20),
#         'VGRF_L7': np.arange(20),
#         'VGRF_L8': np.arange(20),
#         'VGRF_R1': np.arange(20),
#         'VGRF_R2': np.arange(20),
#         'VGRF_R3': np.arange(20),
#         'VGRF_R4': np.arange(20),
#         'VGRF_R5': np.arange(20),
#         'VGRF_R6': np.arange(20),
#         'VGRF_R7': np.arange(20),
#         'VGRF_R8': np.arange(20),
#         'subject_type': ['Control']*20,
#         'label': ['Co']*20
#     })
#     df['time'] = np.arange(20)
#     df = df.set_index('time')
#     data = [df]
#     names = ["dummy"]
#     windows = loader.create_sliding_windows(data, names, window_size=5, step_size=2)
#     assert isinstance(windows, list)
#     assert windows[0]["name"] == "dummy"
#     assert "windows" in windows[0]

# @patch("gaitsetpy.dataset.physionet.requests.get")
# @patch("os.path.exists")
# @patch("os.makedirs")
# @patch("os.listdir")
# @patch("gaitsetpy.dataset.physionet.glob")
# def test_load_data_mocks(glob_mock, listdir_mock, makedirs_mock, exists_mock, requests_get_mock):
#     loader = PhysioNetLoader()
#     exists_mock.return_value = True
#     listdir_mock.return_value = ["GaCo01_01.txt"]
#     glob_mock.return_value = ["/tmp/GaCo01_01.txt"]
#     # Patch pandas.read_csv to return a dummy DataFrame
#     with patch("pandas.read_csv") as read_csv_mock:
#         df = pd.DataFrame({
#             0: np.arange(10),
#             1: np.arange(10),
#             2: np.arange(10),
#             3: np.arange(10),
#             4: np.arange(10),
#             5: np.arange(10),
#             6: np.arange(10),
#             7: np.arange(10),
#             8: np.arange(10),
#             9: np.arange(10),
#             10: np.arange(10),
#             11: np.arange(10),
#             12: np.arange(10),
#             13: np.arange(10),
#             14: np.arange(10),
#             15: np.arange(10),
#             16: np.arange(10),
#         })
#         read_csv_mock.return_value = df
#         data, names = loader.load_data("/tmp")
#         assert isinstance(data, list)
#         assert isinstance(names, list)
#         assert len(data) == 1
#         assert len(names) == 1 