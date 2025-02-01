import unittest
import numpy as np
import pandas as pd
from utils import load_daphnet_data, extract_gait_features
from visualization import plot_thigh_data, plot_shank_data, plot_trunk_data, plot_all_data

class TestDataset(unittest.TestCase):
    def setUp(self):
        """
        Set up test data.
        """
        # Mock dataset
        self.data_dir = "test_data"
        self.daphnet, self.daphnetNames = load_daphnet_data(self.data_dir)
        self.fs = 64  # Sampling frequency

        # Mock thigh, shank, and trunk data
        self.daphnetThigh = [
            pd.DataFrame({
                'thigh_h_fd': np.random.rand(100),
                'thigh_v': np.random.rand(100),
                'thigh_h_l': np.random.rand(100),
                'thigh': np.random.rand(100),
                'annotations': np.random.choice([0, 1, 2], 100)
            })
        ]
        self.daphnetShank = [
            pd.DataFrame({
                'shank_h_fd': np.random.rand(100),
                'shank_v': np.random.rand(100),
                'shank_h_l': np.random.rand(100),
                'shank': np.random.rand(100),
                'annotations': np.random.choice([0, 1, 2], 100)
            })
        ]
        self.daphnetTrunk = [
            pd.DataFrame({
                'trunk_h_fd': np.random.rand(100),
                'trunk_v': np.random.rand(100),
                'trunk_h_l': np.random.rand(100),
                'trunk': np.random.rand(100),
                'annotations': np.random.choice([0, 1, 2], 100)
            })
        ]
        self.daphnetNames = ["S01"]

    def test_load_daphnet_data(self):
        """
        Test that the dataset is loaded correctly.
        """
        self.assertIsInstance(self.daphnet, list)
        self.assertIsInstance(self.daphnetNames, list)
        self.assertEqual(len(self.daphnet), len(self.daphnetNames))
        for df in self.daphnet:
            self.assertIsInstance(df, pd.DataFrame)

    def test_extract_gait_features(self):
        """
        Test that features are extracted correctly.
        """
        features = extract_gait_features(self.daphnet, self.daphnetNames, self.fs)
        self.assertIsInstance(features, list)
        for feature_dict in features:
            self.assertIn("name", feature_dict)
            self.assertIn("features", feature_dict)
            self.assertIn("annotations", feature_dict)

    def test_plot_thigh_data(self):
        """
        Test that the thigh data visualization function works.
        """
        try:
            plot_thigh_data(self.daphnetThigh, self.daphnetNames, 0)
        except Exception as e:
            self.fail(f"plot_thigh_data raised an exception: {e}")

    def test_plot_shank_data(self):
        """
        Test that the shank data visualization function works.
        """
        try:
            plot_shank_data(self.daphnetShank, self.daphnetNames, 0)
        except Exception as e:
            self.fail(f"plot_shank_data raised an exception: {e}")

    def test_plot_trunk_data(self):
        """
        Test that the trunk data visualization function works.
        """
        try:
            plot_trunk_data(self.daphnetTrunk, self.daphnetNames, 0)
        except Exception as e:
            self.fail(f"plot_trunk_data raised an exception: {e}")

    def test_plot_all_data(self):
        """
        Test that the combined visualization function works.
        """
        try:
            plot_all_data(self.daphnetThigh, self.daphnetShank, self.daphnetTrunk, self.daphnetNames, 0)
        except Exception as e:
            self.fail(f"plot_all_data raised an exception: {e}")

if __name__ == "__main__":
    unittest.main()