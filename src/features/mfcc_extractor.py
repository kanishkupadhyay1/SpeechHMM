"""
MFCC Feature Extraction Module for Speech Command Recognition.
Extracts acoustic features using Mel-Frequency Cepstral Coefficients (MFCCs),
including delta and delta-delta coefficients, pre-emphasis, and CMVN normalization.
"""

import io
import os
from typing import Union, Tuple, Optional, Dict, Any
import numpy as np
import librosa
import soundfile as sf
import joblib


class MFCCExtractor:
    """
    Production-grade MFCC Feature Extractor for speech audio.
    Follows standard speech recognition standards:
    - Sample rate: 16,000 Hz
    - Frame duration: 25ms (400 samples)
    - Hop length: 10ms (160 samples)
    - FFT size: 512
    - Base MFCCs: 13
    - Pre-emphasis filter: 0.97
    - Delta + Delta-Delta dynamic features (total 39 dimensions)
    - Cepstral Mean Normalization (CMN)
    """

    def __init__(self,
                 sample_rate: int = 16000,
                 duration: float = 1.0,
                 n_mfcc: int = 13,
                 n_fft: int = 512,
                 hop_length: int = 160,
                 win_length: int = 400,
                 pre_emphasis: float = 0.97,
                 include_deltas: bool = True,
                 use_cmvn: bool = True):
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_samples = int(sample_rate * duration)
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.pre_emphasis = pre_emphasis
        self.include_deltas = include_deltas
        self.use_cmvn = use_cmvn

    def load_audio(self, audio_source: Union[str, bytes, io.BytesIO]) -> Tuple[np.ndarray, int]:
        """
        Load audio from a file path, bytes, or BytesIO buffer.
        Converts stereo to mono and resamples to target sample rate.
        Supports standard WAV, RIFF, MP3, FLAC, OGG, and raw PCM data.
        """
        raw_bytes = None
        if isinstance(audio_source, (bytes, bytearray)):
            raw_bytes = bytes(audio_source)
            audio_buf = io.BytesIO(raw_bytes)
        elif isinstance(audio_source, io.BytesIO):
            audio_buf = audio_source
            raw_bytes = audio_source.getvalue()
        else:
            audio_buf = audio_source

        y = None
        sr = self.sample_rate

        # 1. Try soundfile directly on buffer
        if isinstance(audio_buf, io.BytesIO):
            audio_buf.seek(0)
            try:
                y, sr = sf.read(audio_buf)
            except Exception:
                y = None

        # 2. If soundfile on buffer failed, try writing to a temp file for multi-decoder fallback
        if y is None and raw_bytes is not None and len(raw_bytes) > 0:
            import tempfile
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(raw_bytes)
                    tmp_path = tmp.name

                # Try soundfile on disk file
                try:
                    y, sr = sf.read(tmp_path)
                except Exception:
                    pass

                # Try scipy.io.wavfile
                if y is None:
                    try:
                        import scipy.io.wavfile as wavfile
                        sr_scipy, data = wavfile.read(tmp_path)
                        sr = sr_scipy
                        if data.dtype == np.int16:
                            y = data.astype(np.float32) / 32768.0
                        elif data.dtype == np.int32:
                            y = data.astype(np.float32) / 2147483648.0
                        elif data.dtype == np.uint8:
                            y = (data.astype(np.float32) - 128.0) / 128.0
                        else:
                            y = data.astype(np.float32)
                    except Exception:
                        pass

                # Try wave standard library
                if y is None:
                    try:
                        import wave
                        with wave.open(tmp_path, 'rb') as wf:
                            n_channels = wf.getnchannels()
                            sampwidth = wf.getsampwidth()
                            sr = wf.getframerate()
                            n_frames = wf.getnframes()
                            frames = wf.readframes(n_frames)
                            if sampwidth == 2:
                                data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                                if n_channels > 1:
                                    data = data.reshape(-1, n_channels)
                                y = data
                    except Exception:
                        pass

                # Try librosa.load
                if y is None:
                    try:
                        y, sr = librosa.load(tmp_path, sr=None, mono=True)
                    except Exception:
                        pass
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass

        # 3. If standard decoders failed, check if it's raw PCM 16-bit int data
        if y is None and raw_bytes is not None and len(raw_bytes) >= 160:
            offset = 44 if raw_bytes[:4] == b'RIFF' and len(raw_bytes) > 44 else 0
            pcm_bytes = raw_bytes[offset:]
            if len(pcm_bytes) % 2 != 0:
                pcm_bytes = pcm_bytes[:-1]
            if len(pcm_bytes) >= 320:
                y = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                sr = self.sample_rate

        # 4. If path string was passed
        if y is None and isinstance(audio_source, str) and os.path.isfile(audio_source):
            try:
                y, sr = sf.read(audio_source)
            except Exception:
                y, sr = librosa.load(audio_source, sr=None, mono=True)

        if y is None or len(y) == 0:
            raise ValueError("Unable to decode audio. Please submit a valid WAV audio file.")

        # Convert stereo to mono if necessary
        if y.ndim > 1:
            y = np.mean(y, axis=1)

        # Resample if sample rate doesn't match
        if sr != self.sample_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
            sr = self.sample_rate

        return y.astype(np.float32), sr

    def preprocess_signal(self, y: np.ndarray) -> np.ndarray:
        """
        Apply pre-emphasis, duration standardization, and amplitude normalization.
        """
        # Remove DC offset
        y = y - np.mean(y)

        # Pre-emphasis filter: y[t] = x[t] - alpha * x[t-1]
        if self.pre_emphasis > 0:
            y = np.append(y[0], y[1:] - self.pre_emphasis * y[:-1])

        # Standardize duration to target_samples (e.g., 16000 for 1.0s)
        n_samples = len(y)
        if n_samples < self.target_samples:
            # Center pad with silence
            pad_left = (self.target_samples - n_samples) // 2
            pad_right = self.target_samples - n_samples - pad_left
            y = np.pad(y, (pad_left, pad_right), mode="constant")
        elif n_samples > self.target_samples:
            # Center crop
            start = (n_samples - self.target_samples) // 2
            y = y[start:start + self.target_samples]

        # Amplitude normalization (peak normalization)
        peak = np.max(np.abs(y))
        if peak > 1e-5:
            y = y / peak

        return y

    def extract_features(self, y: np.ndarray) -> np.ndarray:
        """
        Compute MFCC features for preprocessed 1D audio array.
        Returns shape (T, D) where T = time frames, D = feature dimension (13 or 39).
        """
        # Mel spectrogram & MFCC computation
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window="hamming"
        )  # Shape: (n_mfcc, T)

        feature_list = [mfcc]

        if self.include_deltas:
            delta1 = librosa.feature.delta(mfcc, order=1)
            delta2 = librosa.feature.delta(mfcc, order=2)
            feature_list.extend([delta1, delta2])

        # Concatenate along feature axis: (D, T)
        combined = np.vstack(feature_list)

        # Transpose to (T, D) for HMM temporal sequence
        feat_seq = combined.T

        # Cepstral Mean and Variance Normalization (CMVN)
        if self.use_cmvn:
            mean = np.mean(feat_seq, axis=0, keepdims=True)
            std = np.std(feat_seq, axis=0, keepdims=True) + 1e-8
            feat_seq = (feat_seq - mean) / std

        return feat_seq

    def process_file(self, audio_source: Union[str, bytes, io.BytesIO]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete processing pipeline for single audio input.
        Returns:
            processed_audio: (target_samples,) 1D float array
            features: (T, D) 2D float array
        """
        raw_y, sr = self.load_audio(audio_source)
        clean_y = self.preprocess_signal(raw_y)
        features = self.extract_features(clean_y)
        return clean_y, features


def _process_single_record(args):
    """Worker function for parallel extraction."""
    extractor, path, label = args
    try:
        _, feat = extractor.process_file(path)
        return {
            "feat": feat,
            "label": label,
            "length": len(feat),
            "file_path": path,
            "success": True
        }
    except Exception as e:
        return {"error": str(e), "success": False, "file_path": path}


def batch_extract_features(manifest_csv_path: str,
                           output_feature_path: str,
                           extractor: Optional[MFCCExtractor] = None,
                           max_workers: Optional[int] = None) -> Dict[str, Any]:
    """
    Extract MFCC features from manifest CSV in parallel using CPU thread/process pool.
    """
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor, as_completed
    if extractor is None:
        extractor = MFCCExtractor()

    df = pd.read_csv(manifest_csv_path)
    total_samples = len(df)
    print(f"Extracting features for {total_samples} recordings from {manifest_csv_path}...")

    tasks = [(extractor, row["file_path"], row["command"]) for _, row in df.iterrows()]

    all_features = []
    labels = []
    lengths = []
    file_paths = []

    workers = max_workers or min(16, os.cpu_count() or 4)
    print(f"Using {workers} parallel extraction workers...")

    # Process in chunks with ProcessPoolExecutor or ThreadPoolExecutor
    from concurrent.futures import ThreadPoolExecutor
    # Librosa / Soundfile C extensions release GIL during I/O and FFT
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for result in executor.map(_process_single_record, tasks):
            completed += 1
            if result.get("success"):
                all_features.append(result["feat"])
                labels.append(result["label"])
                lengths.append(result["length"])
                file_paths.append(result["file_path"])
            if completed % 2500 == 0 or completed == total_samples:
                print(f"  Processed {completed}/{total_samples} samples ({(completed/total_samples)*100:.1f}%)...")

    os.makedirs(os.path.dirname(output_feature_path), exist_ok=True)
    payload = {
        "features": all_features,
        "labels": labels,
        "lengths": lengths,
        "file_paths": file_paths,
        "extractor_config": {
            "sample_rate": extractor.sample_rate,
            "n_mfcc": extractor.n_mfcc,
            "hop_length": extractor.hop_length,
            "win_length": extractor.win_length,
            "include_deltas": extractor.include_deltas,
            "feature_dim": all_features[0].shape[1] if all_features else 0
        }
    }
    joblib.dump(payload, output_feature_path, compress=3)
    print(f"Cached {len(all_features)} feature sequences to {output_feature_path}")
    return payload


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract MFCC features from manifest")
    parser.add_argument("--manifest", help="Path to manifest CSV")
    parser.add_argument("--output", help="Path to output feature cache")
    parser.add_argument("--workers", type=int, default=None, help="Number of workers")
    args = parser.parse_args()

    if args.manifest and args.output:
        batch_extract_features(args.manifest, args.output, max_workers=args.workers)
    else:
        extractor = MFCCExtractor()
        print("MFCC Extractor initialized with config:", extractor.__dict__)
