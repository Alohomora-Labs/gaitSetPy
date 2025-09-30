'''
UrFall Image/Video Feature Extractor.
Maintainer: @aharshit123456

This module provides a lightweight feature extractor for UrFall depth (PNG16), RGB (PNG), and video (MP4) modalities.
It extracts simple, fast-to-compute features suitable for sanity checks and baseline models.
'''

from typing import List, Dict, Any
import numpy as np
from ..core.base_classes import BaseFeatureExtractor


class UrFallMediaFeatureExtractor(BaseFeatureExtractor):
    """
    UrFall image/video feature extractor.

    Extracts per-window features from sequences of images or decoded video frames:
    - mean_intensity
    - std_intensity
    - motion_mean (if pairwise differences are available)
    - motion_std
    """

    def __init__(self, verbose: bool = False):
        super().__init__(
            name="urfall_media",
            description="Lightweight feature extractor for UrFall depth/RGB/video modalities"
        )
        self.config = {
            'verbose': verbose,
            'use_motion': True,
            'grayscale': True,  # for RGB convert to gray before stats
        }

    def extract_features(self, windows: List[Dict], fs: int, **kwargs) -> List[Dict]:
        self.config.update(kwargs)
        features: List[Dict[str, Any]] = []
        for window in windows:
            name = window.get('name', 'unknown')
            frames = window.get('data', [])
            if not isinstance(frames, list) or len(frames) == 0:
                continue
            # frames is a list of numpy arrays (HxW) or (HxWxC)
            intensities = []
            motions = []
            prev_gray = None
            for f in frames:
                arr = np.array(f)
                if arr.ndim == 3 and self.config['grayscale']:
                    arr = arr.mean(axis=2)
                intensities.append(float(np.mean(arr)))
                if self.config['use_motion']:
                    if prev_gray is not None:
                        diff = np.abs(arr.astype(np.float32) - prev_gray.astype(np.float32))
                        motions.append(float(np.mean(diff)))
                    prev_gray = arr
            fdict: Dict[str, Any] = {'name': name, 'features': {}}
            if intensities:
                fdict['features']['mean_intensity'] = float(np.mean(intensities))
                fdict['features']['std_intensity'] = float(np.std(intensities))
            if motions:
                fdict['features']['motion_mean'] = float(np.mean(motions))
                fdict['features']['motion_std'] = float(np.std(motions))
            features.append(fdict)
        return features

    def get_feature_names(self) -> List[str]:
        return ['mean_intensity', 'std_intensity', 'motion_mean', 'motion_std'] 