"""
Tests for audio preprocessing and MFCC feature extraction.
"""

import os
import glob
import numpy as np
import pytest
from src.features.mfcc_extractor import MFCCExtractor


@pytest.fixture
def sample_wav_path():
    # Find any actual wav file in raw dataset
    wavs = glob.glob(os.path.join("data", "raw", "speech_commands_v0.02", "yes", "*.wav"))
    if not wavs:
        pytest.skip("Dataset wav files not available")
    return wavs[0]


def test_mfcc_extractor_initialization():
    extractor = MFCCExtractor(sample_rate=16000, n_mfcc=13, include_deltas=True)
    assert extractor.sample_rate == 16000
    assert extractor.n_mfcc == 13
    assert extractor.include_deltas is True
    assert extractor.target_samples == 16000


def test_audio_loading_and_shape(sample_wav_path):
    extractor = MFCCExtractor()
    y, sr = extractor.load_audio(sample_wav_path)
    assert sr == 16000
    assert isinstance(y, np.ndarray)
    assert y.ndim == 1


def test_audio_preprocessing(sample_wav_path):
    extractor = MFCCExtractor()
    y, sr = extractor.load_audio(sample_wav_path)
    clean_y = extractor.preprocess_signal(y)
    # Duration standardized to exactly 1.0 second (16000 samples)
    assert len(clean_y) == 16000
    assert np.max(np.abs(clean_y)) <= 1.0 + 1e-4


def test_feature_extraction_dimensions(sample_wav_path):
    extractor = MFCCExtractor(n_mfcc=13, include_deltas=True)
    clean_y, feat = extractor.process_file(sample_wav_path)
    # Feature dimension with 13 MFCC + 13 delta + 13 delta-delta = 39
    assert feat.ndim == 2
    assert feat.shape[1] == 39
    # For 16000 samples with hop_length=160, time frames = 101
    assert 95 <= feat.shape[0] <= 105
    # Check no NaNs or Infs
    assert not np.isnan(feat).any()
    assert not np.isinf(feat).any()
