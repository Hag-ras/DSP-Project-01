# Medical.py (modified for 3-channel display) - removed screenshots, added AI condition identification placeholder
# Requirements:
#   pip install dash pandas numpy scipy plotly pyedflib

import os
import math
import re
import time
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt
import plotly.graph_objs as go
import plotly.io as pio
from dash import Dash, dcc, html, Input, Output, State, callback_context, no_update

# Try importing pyedflib for EDF reading (EEG). If missing, app will instruct user.
try:
    import pyedflib

    PYEDFLIB_AVAILABLE = True
except Exception:
    PYEDFLIB_AVAILABLE = False


import base64
import io
import time
from datetime import datetime
from PIL import Image, ImageOps

from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.layers import Input as KerasInput
from keras.models import load_model,Sequential
import numpy as np
# Screenshot capability check
try:
    import plotly.io as pio
    import kaleido
    SCREENSHOT_AVAILABLE = True
    print("[Screenshot] Plotly screenshot capabilities available")
except ImportError as e:
    SCREENSHOT_AVAILABLE = False
    print(f"[Screenshot] Warning: {e}")
    print("[Screenshot] Install with: pip install kaleido")
# ---------- Configuration / Limits ----------
MAX_EEG_SUBJECTS = 150  # limit EEG subjects loaded
DEFAULT_MAX_EEG_SECONDS = 60  # read at most this many seconds per EDF to save memory (None to read whole file)
DEFAULT_MAX_EEG_SAMPLES = None  # computed from seconds & fs when reading



import plotly.io as pio
from PIL import Image
import tempfile

def capture_graph_screenshot(figure):
    """
    Capture a screenshot of the current plotly figure and return as bytes.
    """
    try:
        # Convert figure to image bytes
        img_bytes = pio.to_image(figure, format="png", width=800, height=600, engine="kaleido")
        return img_bytes
    except Exception as e:
        print(f"[Screenshot] Error capturing graph: {e}")
        return None

# ---------- AI Model Placeholder ----------
class ConditionIdentificationModel:
    """
    RNN-based ECG condition identification model.
    Analyzes raw ECG signal data from .dat files using a trained RNN model.
    """

    def __init__(
            self,
            model_path="ECG.weights.h5",
            labels_path="labels.txt",
            sequence_length=1000,  # Number of samples to analyze at once
            sampling_rate=250,
            confidence_threshold=0.5,
    ):
        # Model configuration
        self.model_path = model_path
        self.labels_path = labels_path
        self.sequence_length = sequence_length
        self.sampling_rate = sampling_rate
        self.confidence_threshold = confidence_threshold

        # Model state
        self._model = None
        self._labels = []
        self._model_loaded = False

        # Performance tracking
        self._last_analysis_time = None
        self._total_predictions = 0

        # Signal preprocessing parameters
        self.target_fs = 250  # Target sampling frequency
        self.filter_params = {
            'lowcut': 0.5,  # High-pass filter cutoff
            'highcut': 40.0,  # Low-pass filter cutoff
            'order': 4  # Filter order
        }

    def load_model(self):
        """
        Load the RNN model. First try to load as complete model, then try weights-only.
        Returns True if successful, False otherwise.
        """
        if self._model_loaded:
            print("[RNN Model] Model already loaded")
            return True

        try:
            # Check if files exist
            if not os.path.exists(self.model_path):
                print(f"[RNN Model] Model file not found: {self.model_path}")
                return False

            if not os.path.exists(self.labels_path):
                print(f"[RNN Model] Labels file not found: {self.labels_path}")
                return False

            # Load class labels first
            print(f"[RNN Model] Loading class labels from: {self.labels_path}")
            with open(self.labels_path, "r", encoding="utf-8") as f:
                self._labels = [line.strip() for line in f.readlines() if line.strip()]

            if not self._labels:
                print("[RNN Model] Error: No labels found in labels file")
                return False

            num_classes = len(self._labels)
            print(f"[RNN Model] Found {num_classes} classes")

            # Method 1: Try to load as complete model (if it's actually a full .h5 model)
            try:
                print(f"[RNN Model] Attempting to load complete model from: {self.model_path}")
                self._model = load_model(self.model_path, compile=False)
                print(f"[RNN Model] Successfully loaded complete model")
                print(f"[RNN Model] Model input shape: {self._model.input_shape}")
                print(f"[RNN Model] Model output shape: {self._model.output_shape}")

                # Update sequence_length based on loaded model
                if hasattr(self._model, 'input_shape') and len(self._model.input_shape) >= 2:
                    self.sequence_length = self._model.input_shape[1] or self.sequence_length
                    print(f"[RNN Model] Updated sequence_length to: {self.sequence_length}")

                self._model_loaded = True
                return True

            except Exception as e1:
                print(f"[RNN Model] Complete model loading failed: {e1}")
                print("[RNN Model] Trying to build architecture and load weights...")

                # Method 2: Build architecture and load weights
                try:
                    # Try different common architectures that might match your weights
                    architectures = [
                        # Architecture 1: Simple LSTM
                        lambda: Sequential([
                            KerasInput(shape=(self.sequence_length, 1)),
                            LSTM(128, return_sequences=True),
                            LSTM(64, return_sequences=True),
                            LSTM(32, return_sequences=False),
                            Dense(64, activation='relu'),
                            Dense(num_classes, activation='softmax')
                        ]),
                        # Architecture 2: LSTM with BatchNorm
                        lambda: Sequential([
                            KerasInput(shape=(self.sequence_length, 1)),
                            LSTM(128, return_sequences=True),
                            BatchNormalization(),
                            LSTM(64, return_sequences=True),
                            BatchNormalization(),
                            LSTM(32, return_sequences=False),
                            Dense(64, activation='relu'),
                            Dense(num_classes, activation='softmax')
                        ]),
                        # Architecture 3: LSTM with Dropout
                        lambda: Sequential([
                            KerasInput(shape=(self.sequence_length, 1)),
                            LSTM(128, return_sequences=True),
                            Dropout(0.2),
                            LSTM(64, return_sequences=True),
                            Dropout(0.2),
                            LSTM(32, return_sequences=False),
                            Dropout(0.2),
                            Dense(64, activation='relu'),
                            Dense(num_classes, activation='softmax')
                        ])
                    ]

                    # Try each architecture
                    for i, arch_func in enumerate(architectures):
                        try:
                            print(f"[RNN Model] Trying architecture {i + 1}...")
                            self._model = arch_func()
                            self._model.compile(optimizer='adam', loss='categorical_crossentropy')

                            # Try to load weights
                            self._model.load_weights(self.model_path)
                            print(f"[RNN Model] Successfully loaded weights with architecture {i + 1}")
                            self._model_loaded = True
                            return True

                        except Exception as arch_error:
                            print(f"[RNN Model] Architecture {i + 1} failed: {arch_error}")
                            continue

                    # If all architectures failed, provide helpful error message
                    print("[RNN Model] All common architectures failed.")
                    print("[RNN Model] Your weights file might need a specific architecture.")
                    print("[RNN Model] Options:")
                    print("  1. Provide the complete .h5 model file (not just weights)")
                    print("  2. Provide the original model architecture code")
                    print("  3. Use a model.save() file instead of model.save_weights()")

                    return False

                except Exception as e2:
                    print(f"[RNN Model] Weight loading failed: {e2}")
                    return False

        except Exception as e:
            print(f"[RNN Model] Failed to load model: {e}")
            self._model_loaded = False
            return False

    def _preprocess_signal(self, signal_data, original_fs=None):
        """
        Preprocess ECG signal for RNN input.

        Args:
            signal_data: Raw ECG signal data (numpy array)
            original_fs: Original sampling frequency (if different from target)

        Returns:
            Preprocessed signal ready for model input
        """
        try:
            signal = np.asarray(signal_data, dtype=np.float32)

            if len(signal) == 0:
                raise ValueError("Empty signal data")

            # Remove NaN values
            if np.any(np.isnan(signal)):
                signal = signal[~np.isnan(signal)]
                print(f"[RNN Model] Removed {np.sum(np.isnan(signal_data))} NaN values")

            # Resample if necessary
            if original_fs and original_fs != self.target_fs:
                from scipy import signal as sp_signal
                num_samples = int(len(signal) * self.target_fs / original_fs)
                signal = sp_signal.resample(signal, num_samples)
                print(f"[RNN Model] Resampled from {original_fs}Hz to {self.target_fs}Hz")

            # Apply bandpass filter
            from scipy.signal import butter, filtfilt
            nyquist = self.target_fs / 2
            low = self.filter_params['lowcut'] / nyquist
            high = self.filter_params['highcut'] / nyquist

            if high >= 1.0:
                high = 0.99

            b, a = butter(self.filter_params['order'], [low, high], btype='band')
            signal = filtfilt(b, a, signal)

            # Normalize signal (z-score normalization)
            signal_mean = np.mean(signal)
            signal_std = np.std(signal)
            if signal_std > 0:
                signal = (signal - signal_mean) / signal_std

            # Extract sequence of target length
            if len(signal) >= self.sequence_length:
                # Take the most recent samples
                signal = signal[-self.sequence_length:]
            else:
                # Pad with zeros if signal is too short
                padded_signal = np.zeros(self.sequence_length)
                padded_signal[:len(signal)] = signal
                signal = padded_signal
                print(f"[RNN Model] Padded signal from {len(signal_data)} to {self.sequence_length} samples")

            # Reshape for RNN input: (1, sequence_length, 1)
            signal = signal.reshape(1, self.sequence_length, 1)

            print(f"[RNN Model] Signal preprocessed: shape {signal.shape}")
            return signal

        except Exception as e:
            print(f"[RNN Model] Signal preprocessing error: {e}")
            raise

    def analyze_signal_data(self, signal_data, sampling_rate=250, top_k=5):
        """
        Analyze ECG signal data and predict medical conditions.

        Args:
            signal_data: Raw ECG signal (numpy array or list)
            sampling_rate: Sampling frequency of the input signal
            top_k: Number of top predictions to return

        Returns:
            Dictionary with prediction results
        """
        analysis_start = time.time()

        # Ensure model is loaded
        if not self._model_loaded:
            if not self.load_model():
                return {
                    "error": "Failed to load RNN model. Check that ECG.weights.h5 and labels.txt exist.",
                    "timestamp": datetime.now().isoformat()
                }

        try:
            # Validate input
            if signal_data is None or len(signal_data) == 0:
                return {
                    "error": "No signal data provided",
                    "timestamp": datetime.now().isoformat()
                }

            # Preprocess signal
            print("[RNN Model] Preprocessing signal data...")
            processed_signal = self._preprocess_signal(signal_data, sampling_rate)

            # Run model prediction
            print("[RNN Model] Running RNN inference...")
            inference_start = time.time()

            # Suppress TensorFlow logging
            import logging
            tf_logger = logging.getLogger('tensorflow')
            original_level = tf_logger.level
            tf_logger.setLevel(logging.ERROR)

            try:
                predictions = self._model.predict(processed_signal, verbose=0)
            finally:
                tf_logger.setLevel(original_level)

            inference_time = time.time() - inference_start

            # Process predictions
            probabilities = predictions[0]  # Remove batch dimension

            # Get top-k predictions
            top_k = min(max(1, int(top_k)), len(probabilities))
            top_indices = np.argsort(probabilities)[::-1][:top_k]

            prediction_results = []
            for idx in top_indices:
                class_index = int(idx)
                confidence = float(probabilities[idx])

                # Get label name
                if class_index < len(self._labels):
                    label = self._labels[class_index]
                else:
                    label = f"Class_{class_index}"

                prediction_results.append({
                    "index": class_index,
                    "label": label,
                    "confidence": confidence,
                    "confidence_percent": round(confidence * 100, 1)
                })

            # Calculate timing
            total_time = time.time() - analysis_start
            self._last_analysis_time = total_time
            self._total_predictions += 1

            # Determine prediction quality
            top_confidence = prediction_results[0]["confidence"] if prediction_results else 0.0
            if top_confidence >= 0.8:
                quality = "High"
            elif top_confidence >= 0.6:
                quality = "Medium"
            else:
                quality = "Low"

            print(f"[RNN Model] Analysis complete: top confidence {top_confidence:.3f}")

            return {
                "success": True,
                "predictions": prediction_results,
                "raw_probabilities": probabilities.tolist(),
                "prediction_quality": quality,
                "top_confidence": top_confidence,
                "inference_time_s": round(inference_time, 4),
                "total_time_s": round(total_time, 4),
                "sequence_length": self.sequence_length,
                "sampling_rate": sampling_rate,
                "signal_length": len(signal_data),
                "timestamp": datetime.now().isoformat(),
                "prediction_count": self._total_predictions
            }

        except Exception as e:
            error_msg = f"RNN analysis failed: {str(e)}"
            print(f"[RNN Model] {error_msg}")
            return {
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            }

    def analyze_patient_data(self, patient_data, channel_name="signal_1", top_k=5):
        """
        Analyze ECG data from a patient DataFrame.

        Args:
            patient_data: DataFrame with ECG data
            channel_name: Name of the signal column to analyze
            top_k: Number of top predictions to return

        Returns:
            Dictionary with prediction results
        """
        try:
            if channel_name not in patient_data.columns:
                available_channels = [col for col in patient_data.columns if col.startswith('signal_')]
                if not available_channels:
                    return {"error": "No signal channels found in patient data"}
                channel_name = available_channels[0]
                print(f"[RNN Model] Using channel: {channel_name}")

            # Extract signal data
            signal_data = patient_data[channel_name].values

            # Get sampling rate from data if available
            if 'time' in patient_data.columns and len(patient_data) > 1:
                time_diff = patient_data['time'].iloc[1] - patient_data['time'].iloc[0]
                estimated_fs = 1.0 / time_diff if time_diff > 0 else self.sampling_rate
            else:
                estimated_fs = self.sampling_rate

            return self.analyze_signal_data(signal_data, estimated_fs, top_k)

        except Exception as e:
            return {"error": f"Patient data analysis failed: {e}"}

    def get_model_info(self):
        """Get information about the loaded model."""
        return {
            "model_loaded": self._model_loaded,
            "model_path": self.model_path,
            "labels_path": self.labels_path,
            "num_classes": len(self._labels),
            "sequence_length": self.sequence_length,
            "sampling_rate": self.sampling_rate,
            "confidence_threshold": self.confidence_threshold,
            "total_predictions": self._total_predictions,
            "last_analysis_time": self._last_analysis_time,
            "available_labels": self._labels[:10] if len(self._labels) > 10 else self._labels
        }

    def is_ready(self):
        """Check if the model is loaded and ready for predictions."""
        return self._model_loaded and self._model is not None

    def get_supported_conditions(self, signal_type="ECG"):
        """Return list of conditions the model can identify."""
        if self._model_loaded:
            return self._labels
        else:
            return []

    # Legacy compatibility method
    def analyze_signal(self, signal_data, signal_type="ECG", sampling_rate=250):
        """Legacy method for backwards compatibility."""
        result = self.analyze_signal_data(signal_data, sampling_rate, top_k=3)

        if "error" in result:
            return {
                "condition": "Analysis Error",
                "confidence": 0.0,
                "features": {},
                "timestamp": result["timestamp"]
            }

        predictions = result.get("predictions", [])
        if predictions:
            top_pred = predictions[0]
            return {
                "condition": top_pred["label"],
                "confidence": top_pred["confidence"],
                "features": {
                    "signal_length": result.get("signal_length", 0),
                    "sampling_rate": sampling_rate,
                    "prediction_quality": result.get("prediction_quality", "Unknown")
                },
                "timestamp": result["timestamp"]
            }
        else:
            return {
                "condition": "No Prediction",
                "confidence": 0.0,
                "features": {},
                "timestamp": result["timestamp"]
            }

# Initialize global AI model instance
AI_MODEL = ConditionIdentificationModel()

#---------- Channel Processing Functions ----------
def derive_third_ecg_channel(sig1, sig2, method="difference"):
    """
    Derive a third ECG channel from two existing channels.

    Args:
        sig1, sig2: numpy arrays of the two ECG channels
        method: "difference", "sum", or "orthogonal"

    Returns:
        numpy array of the derived third channel
    """
    if len(sig1) != len(sig2):
        min_len = min(len(sig1), len(sig2))
        sig1, sig2 = sig1[:min_len], sig2[:min_len]

    if method == "difference":
        # Lead III = Lead II - Lead I (similar to standard ECG lead derivation)
        derived = sig2 - sig1
        print(f"[Channel Derivation] Created third ECG channel using difference method (Lead2 - Lead1)")
    elif method == "sum":
        # Sum of the two leads
        derived = (sig1 + sig2) / 2
        print(f"[Channel Derivation] Created third ECG channel using sum method ((Lead1 + Lead2)/2)")
    elif method == "orthogonal":
        # Create an orthogonal lead using Gram-Schmidt-like process
        # This approximates a perpendicular view
        dot_product = np.dot(sig1, sig2) / (np.linalg.norm(sig1) * np.linalg.norm(sig2))
        derived = sig2 - dot_product * sig1
        print(f"[Channel Derivation] Created third ECG channel using orthogonal method")
    else:
        derived = sig2 - sig1  # Default to difference
        print(f"[Channel Derivation] Created third ECG channel using default difference method")

    return derived


def get_display_channels(patient, dataset_type, show_all_channels=False):
    """
    Get the channels to display based on dataset type and available channels.

    Args:
        patient: patient data dict
        dataset_type: "ECG" or "EEG"
        show_all_channels: whether to show all available channels

    Returns:
        list of channel names to display
    """
    if "ecg" not in patient or patient["ecg"] is None:
        return []

    available_channels = [c for c in patient["ecg"].columns if c.startswith("signal_")]

    if dataset_type == "ECG":
        if len(available_channels) == 0:
            return []
        elif len(available_channels) == 1:
            # Only one channel - duplicate it for now, will handle in visualization
            return [available_channels[0]]
        elif len(available_channels) == 2:
            # Two channels - will derive third in visualization
            return available_channels
        elif len(available_channels) >= 3:
            if show_all_channels:
                return available_channels
            else:
                # Show first 3 channels by default
                return available_channels[:3]

    else:  # EEG
        if len(available_channels) <= 3:
            return available_channels
        else:
            if show_all_channels:
                return available_channels
            else:
                # Show first 3 channels by default for EEG
                return available_channels[:3]

    return available_channels


def process_patient_channels(patient, dataset_type, show_all_channels=False):
    """
    Process patient data to ensure 3 channels for display.
    For ECG: derive third channel if only 2 exist.
    For EEG: limit to 3 main channels unless show_all is True.

    Returns:
        dict with processed channel data and metadata
    """
    if "ecg" not in patient or patient["ecg"] is None:
        return {"channels": [], "derived_info": "No data available"}

    ecg_df = patient["ecg"].copy()
    available_channels = [c for c in ecg_df.columns if c.startswith("signal_")]

    result = {
        "channels": [],
        "derived_info": "",
        "original_count": len(available_channels)
    }

    if dataset_type == "ECG":
        if len(available_channels) == 0:
            result["derived_info"] = "No ECG channels available"
            return result
        elif len(available_channels) == 1:
            # Single channel - create 3 versions for visualization
            ch = available_channels[0]
            result["channels"] = [ch, f"{ch}_copy", f"{ch}_inverted"]
            # Add copies to dataframe
            ecg_df[f"{ch}_copy"] = ecg_df[ch]
            ecg_df[f"{ch}_inverted"] = -ecg_df[ch]  # Inverted for different perspective
            result["derived_info"] = f"Single channel {ch} displayed with copy and inverted version"
        elif len(available_channels) == 2:
            # Two channels - derive third
            ch1, ch2 = available_channels[0], available_channels[1]
            derived_ch = f"derived_{ch1}_{ch2}"
            derived_signal = derive_third_ecg_channel(ecg_df[ch1].values, ecg_df[ch2].values, method="difference")
            ecg_df[derived_ch] = derived_signal
            result["channels"] = [ch1, ch2, derived_ch]
            result["derived_info"] = f"Third channel '{derived_ch}' derived from {ch1} - {ch2}"
        else:
            # Three or more channels
            if show_all_channels:
                result["channels"] = available_channels
                result["derived_info"] = f"Showing all {len(available_channels)} channels"
            else:
                result["channels"] = available_channels[:3]
                result["derived_info"] = f"Showing main 3 channels (out of {len(available_channels)} available)"

    else:  # EEG
        if len(available_channels) <= 3:
            result["channels"] = available_channels
            result["derived_info"] = f"Showing all {len(available_channels)} EEG channels"
        else:
            if show_all_channels:
                result["channels"] = available_channels
                result["derived_info"] = f"Showing all {len(available_channels)} EEG channels"
            else:
                result["channels"] = available_channels[:3]
                result["derived_info"] = f"Showing main 3 EEG channels (out of {len(available_channels)} available)"

    # Update patient data with processed dataframe
    patient["ecg"] = ecg_df
    result["processed_df"] = ecg_df

    return result


# ---------- Utilities ----------
def parse_num(token, default=None):
    if token is None:
        return default
    token = str(token).strip()
    if token == "":
        return default
    try:
        return float(token)
    except:
        pass
    if '/' in token:
        parts = token.split('/')
        for p in parts:
            p = p.strip()
            m = re.search(r'[-+]?\d+(\.\d+)?', p)
            if m:
                try:
                    return float(m.group(0))
                except:
                    continue
    m = re.search(r'[-+]?\d+(\.\d+)?', token)
    if m:
        try:
            return float(m.group(0))
        except:
            return default
    return default


def find_dataset_directory(dataset_type, root="."):
    """
    Updated dataset directory finder that looks for patient-organized structures.
    """
    if dataset_type == "ECG":
        candidates = [
            os.path.join(os.getcwd(), "data", "ptbdb"),
            os.path.join(os.getcwd(), "ptbdb"),
            os.path.join(os.getcwd(), "qtdb_data", "physionet.org", "files", "qtdb", "1.0.0"),
            os.path.join(os.getcwd(), "qtdb"),
            os.path.join(os.getcwd(), "qtdb_data"),
            os.path.join(os.getcwd(), "1.0.0"),
            os.path.join(os.getcwd(), "qtdb", "1.0.0"),
            os.path.join(os.getcwd(), "qtdb-1.0.0"),
        ]

        # Check for patient-organized structure first
        for d in candidates:
            if os.path.isdir(d):
                # Look for patient directories
                patient_dirs = find_patient_directories(d)
                if patient_dirs:
                    return d

                # Fallback: look for .hea files directly
                for rootd, _, files in os.walk(d):
                    for f in files:
                        if f.lower().endswith('.hea'):
                            return d
    else:
        # EEG logic remains the same
        candidates = [
            os.path.join(os.getcwd(), "ASZED-153"),
            os.path.join(os.getcwd(), "ASZED_153"),
            os.path.join(os.getcwd(), "aszed-153"),
            os.path.join(os.getcwd(), "eeg_data"),
        ]

        for d in candidates:
            if os.path.isdir(d):
                for rootd, _, files in os.walk(d):
                    for f in files:
                        if f.lower().endswith('.edf'):
                            return d

    # Fallback: scan root
    for rootd, _, files in os.walk(root):
        for f in files:
            ext = '.hea' if dataset_type == "ECG" else '.edf'
            if f.lower().endswith(ext):
                return rootd
    return None


def get_subject_id_from_path(path):
    """
    Heuristic to extract subject id from file path.
    """
    if not path:
        return None
    m = re.search(r"subject[_\-]?(\d+)", path, flags=re.IGNORECASE)
    if m:
        return f"subject_{int(m.group(1))}"
    parts = re.split(r"[\\/]+", path)
    if len(parts) >= 2:
        for p in reversed(parts[:-1]):
            mm = re.match(r"^(\d+)$", p)
            if mm:
                return f"subject_{int(mm.group(1))}"
            mm2 = re.match(r"^sub[_\-]?(\d+)$", p, flags=re.IGNORECASE)
            if mm2:
                return f"subject_{int(mm2.group(1))}"
        return parts[-2]
    return os.path.basename(path)


def find_all_data_files(root_dir, file_extension):
    """
    Recursively find all data files with given extension in directory tree.
    Groups files by their immediate parent directory.

    Args:
        root_dir: Root directory to search
        file_extension: '.dat' for ECG or '.edf' for EEG

    Returns:
        Dictionary mapping parent_dir -> list of file paths
    """
    grouped_files = {}

    for root, dirs, files in os.walk(root_dir):
        matching_files = [f for f in files if f.lower().endswith(file_extension)]

        if matching_files:
            # Use the immediate parent directory as the group key
            parent_key = os.path.basename(root) or root
            full_paths = [os.path.join(root, f) for f in sorted(matching_files)]

            if parent_key not in grouped_files:
                grouped_files[parent_key] = []
            grouped_files[parent_key].extend(full_paths)

    return grouped_files


def concatenate_ecg_files(file_paths, max_samples=None):
    """
    Vertically concatenate multiple ECG .dat/.hea file pairs.

    Args:
        file_paths: List of .dat or .hea file paths to concatenate
        max_samples: Maximum samples to read (None for all)

    Returns:
        (combined_df, combined_header) or (None, None) on failure
    """
    combined_df = None
    combined_header = None
    total_samples = 0

    for file_path in file_paths:
        # Get corresponding .hea and .dat files
        if file_path.endswith('.hea'):
            hea_path = file_path
            dat_path = file_path.replace('.hea', '.dat')
        elif file_path.endswith('.dat'):
            dat_path = file_path
            hea_path = file_path.replace('.dat', '.hea')
        else:
            continue

        if not os.path.exists(hea_path) or not os.path.exists(dat_path):
            print(f"[concatenate_ecg_files] Missing pair for {file_path}")
            continue

        # Read header and data
        header = read_header_file(hea_path)
        if header is None:
            print(f"[concatenate_ecg_files] Failed to read header: {hea_path}")
            continue

        df = read_dat_file(dat_path, header, max_samples=max_samples)
        if df is None:
            print(f"[concatenate_ecg_files] Failed to read data: {dat_path}")
            continue

        # First file - initialize
        if combined_df is None:
            combined_df = df.copy()
            combined_header = header.copy()
            combined_header['source_files'] = [os.path.basename(file_path)]
            total_samples = len(df)
        else:
            # Subsequent files - concatenate
            # Ensure same number of signal columns
            common_signals = [col for col in df.columns if col.startswith('signal_') and col in combined_df.columns]

            if not common_signals:
                print(f"[concatenate_ecg_files] No matching signal columns in {file_path}")
                continue

            # Adjust time column to continue from previous data
            last_time = combined_df['time'].iloc[-1]
            fs = header.get('sampling_frequency', 250.0)
            time_increment = 1.0 / fs
            df['time'] = df['time'] + last_time + time_increment

            # Concatenate only matching columns
            cols_to_concat = ['time'] + common_signals
            combined_df = pd.concat([combined_df[cols_to_concat], df[cols_to_concat]],
                                    ignore_index=True, axis=0)

            combined_header['source_files'].append(os.path.basename(file_path))
            total_samples += len(df)

    if combined_df is not None:
        combined_header['num_samples'] = total_samples
        combined_header['concatenated_files'] = len(combined_header['source_files'])
        print(f"[concatenate_ecg_files] Combined {len(combined_header['source_files'])} files, "
              f"total samples: {total_samples}")

    return combined_df, combined_header


def concatenate_eeg_files(file_paths, max_samples=None):
    """
    Vertically concatenate multiple EEG .edf files.

    Args:
        file_paths: List of .edf file paths to concatenate
        max_samples: Maximum samples to read per file (None for all)

    Returns:
        (combined_df, combined_header) or (None, None) on failure
    """
    if not PYEDFLIB_AVAILABLE:
        print("[concatenate_eeg_files] pyedflib not available")
        return None, None

    combined_df = None
    combined_header = None
    total_samples = 0

    for file_path in file_paths:
        df, header = read_edf_file(file_path, max_samples=max_samples)

        if df is None:
            print(f"[concatenate_eeg_files] Failed to read: {file_path}")
            continue

        # First file - initialize
        if combined_df is None:
            combined_df = df.copy()
            combined_header = header.copy()
            combined_header['source_files'] = [os.path.basename(file_path)]
            total_samples = len(df)
        else:
            # Subsequent files - concatenate
            # Ensure same number of signal columns
            common_signals = [col for col in df.columns if col.startswith('signal_') and col in combined_df.columns]

            if not common_signals:
                print(f"[concatenate_eeg_files] No matching signal columns in {file_path}")
                continue

            # Adjust time column to continue from previous data
            last_time = combined_df['time'].iloc[-1]
            fs = header.get('sampling_frequency', 250)
            time_increment = 1.0 / fs
            df['time'] = df['time'] + last_time + time_increment

            # Concatenate only matching columns
            cols_to_concat = ['time'] + common_signals
            combined_df = pd.concat([combined_df[cols_to_concat], df[cols_to_concat]],
                                    ignore_index=True, axis=0)

            combined_header['source_files'].append(os.path.basename(file_path))
            total_samples += len(df)

    if combined_df is not None:
        combined_header['num_samples'] = total_samples
        combined_header['concatenated_files'] = len(combined_header['source_files'])
        print(f"[concatenate_eeg_files] Combined {len(combined_header['source_files'])} files, "
              f"total samples: {total_samples}")

    return combined_df, combined_header

def read_header_file(path):
    """Read .hea header (robust)."""
    if not os.path.exists(path):
        return None
    with open(path, "r", errors="ignore") as fh:
        lines = [ln.strip() for ln in fh.readlines() if ln.strip() != ""]
    if not lines:
        return None
    first = lines[0].split()
    record_name = first[0] if len(first) >= 1 else None
    num_signals = parse_num(first[1], default=2) if len(first) >= 2 else 2
    fs = parse_num(first[2], default=250.0) if len(first) >= 3 else 250.0
    num_samples = parse_num(first[3], default=225000) if len(first) >= 4 else 225000
    try:
        num_signals = int(num_signals)
    except:
        num_signals = int(max(1, math.floor(num_signals))) if num_signals else 2
    try:
        fs = float(fs)
    except:
        fs = 250.0
    try:
        num_samples = int(num_samples)
    except:
        num_samples = int(225000)
    signals_raw = []
    if len(lines) > 1:
        for ln in lines[1:]:
            signals_raw.append(ln.split())
    return {
        "record_name": record_name,
        "num_signals": num_signals,
        "sampling_frequency": fs,
        "num_samples": num_samples,
        "signals_raw": signals_raw
    }


def read_dat_file(dat_path, header_info, max_samples=None):
    """Read MIT/PhysioNet .dat interleaved int16. Return pandas DataFrame or None on failure."""
    if not os.path.exists(dat_path) or header_info is None:
        return None
    try:
        raw = np.fromfile(dat_path, dtype=np.int16)
        n_signals = max(1, int(header_info.get("num_signals", 2)))
        total_samples = raw.shape[0] // n_signals
        if total_samples <= 0:
            return None
        raw = raw[: total_samples * n_signals]
        mat = raw.reshape((total_samples, n_signals))
        gains = np.ones(n_signals) * 200.0
        for i in range(min(n_signals, len(header_info.get("signals_raw", [])))):
            parts = header_info["signals_raw"][i]
            if len(parts) >= 3:
                g = parse_num(parts[2], default=None)
                if g and g > 0:
                    gains[i] = g
        cols = [f"signal_{i + 1}" for i in range(n_signals)]
        df = pd.DataFrame(mat[:, :n_signals].astype(float) / gains[:n_signals], columns=cols)
        fs = header_info.get("sampling_frequency", 250.0)
        df.insert(0, "time", np.arange(df.shape[0]) / float(fs))
        if max_samples is not None:
            df = df.iloc[: int(max_samples)].reset_index(drop=True)
        return df
    except Exception as e:
        print(f"[read_dat_file] error reading {dat_path}: {e}")
        return None


def find_patient_directories(data_dir):
    """
    Find all patient directories in the dataset.
    Looks for directories named like 'patient001', 'patient002', etc.
    """
    patient_dirs = []

    if not os.path.isdir(data_dir):
        return patient_dirs

    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path):
            # Check if it looks like a patient directory
            if (item.lower().startswith('patient') or
                    item.lower().startswith('subject') or
                    item.isdigit()):  # Some datasets use just numbers
                patient_dirs.append(item_path)

    return sorted(patient_dirs)


def get_patient_records(patient_dir):
    """
    Get all ECG records (hea/dat file pairs) for a specific patient.
    Returns list of record base names (without extensions).
    """
    if not os.path.isdir(patient_dir):
        return []

    records = []
    files = os.listdir(patient_dir)

    # Find all .hea files and extract base names
    for f in files:
        if f.lower().endswith('.hea'):
            base_name = os.path.splitext(f)[0]
            # Check if corresponding .dat file exists
            dat_file = os.path.join(patient_dir, f"{base_name}.dat")
            if os.path.exists(dat_file):
                records.append({
                    'base_name': base_name,
                    'hea_path': os.path.join(patient_dir, f),
                    'dat_path': dat_file
                })

    return records


def load_patient_record(record_info, max_samples=None):
    """
    Load a single ECG record (hea + dat file pair).
    Returns DataFrame with all channels from that record.
    """
    hea_path = record_info['hea_path']
    dat_path = record_info['dat_path']

    # Read header
    header = read_header_file(hea_path)
    if header is None:
        return None, None

    # Read data
    df = read_dat_file(dat_path, header, max_samples=max_samples)
    if df is None:
        return None, None

    # Add record identifier to column names to avoid conflicts when combining
    base_name = record_info['base_name']
    signal_cols = [c for c in df.columns if c.startswith('signal_')]
    rename_dict = {}
    for i, col in enumerate(signal_cols):
        rename_dict[col] = f"signal_{base_name}_{i + 1}"

    df.rename(columns=rename_dict, inplace=True)

    return df, header


def combine_patient_records(patient_dir, max_samples=None, max_records_per_patient=None):
    """
    Combine all ECG records for a single patient into one DataFrame.
    Each record contributes its channels with unique names.
    """
    records = get_patient_records(patient_dir)
    if not records:
        return None, None

    # Limit records per patient if specified
    if max_records_per_patient:
        records = records[:max_records_per_patient]

    combined_df = None
    combined_header = None
    total_channels = 0

    for i, record_info in enumerate(records):
        try:
            df, header = load_patient_record(record_info, max_samples)
            if df is None:
                print(f"[combine_patient_records] Failed to load record {record_info['base_name']}")
                continue

            if combined_df is None:
                # First record - use as base
                combined_df = df.copy()
                combined_header = header.copy()
                combined_header['records'] = [record_info['base_name']]
            else:
                # Additional records - merge
                if len(df) != len(combined_df):
                    # Handle different lengths by taking minimum
                    min_len = min(len(df), len(combined_df))
                    df = df.iloc[:min_len].reset_index(drop=True)
                    combined_df = combined_df.iloc[:min_len].reset_index(drop=True)

                # Add signal columns from this record
                signal_cols = [c for c in df.columns if c.startswith('signal_')]
                for col in signal_cols:
                    combined_df[col] = df[col].values

                combined_header['records'].append(record_info['base_name'])

            # Count channels added
            record_channels = len([c for c in df.columns if c.startswith('signal_')])
            total_channels += record_channels

        except Exception as e:
            print(f"[combine_patient_records] Error processing record {record_info['base_name']}: {e}")
            continue

    if combined_df is not None:
        # Update header with combined info
        combined_header['num_signals'] = len([c for c in combined_df.columns if c.startswith('signal_')])
        combined_header['combined_records'] = len(records)
        combined_header['total_channels'] = total_channels

        print(
            f"[combine_patient_records] Combined {len(records)} records into {combined_header['num_signals']} total channels")

    return combined_df, combined_header


def read_edf_file(edf_path, max_samples=None, attempts=8):
    """
    Read EDF file using pyedflib. Returns (df, header) or (None, None) and prints error.
    """
    if not PYEDFLIB_AVAILABLE:
        print("pyedflib not installed. Please: pip install pyedflib")
        return None, None

    last_exc = None
    backoff = 0.05
    for attempt in range(attempts):
        try:
            f = pyedflib.EdfReader(edf_path)
            try:
                try:
                    n_signals = int(f.signals_in_file)
                except Exception:
                    n_signals = int(getattr(f, "signals_in_file", 0) or 0)
                if n_signals <= 0:
                    raise ValueError("No signals in EDF")
                nsamps = f.getNSamples()
                if isinstance(nsamps, (list, tuple, np.ndarray)):
                    min_samples = int(min(nsamps))
                else:
                    min_samples = int(nsamps)
                use_samples = min_samples if max_samples is None else min(min_samples, int(max_samples))
                fs = None
                try:
                    fs = int(f.getSampleFrequency(0))
                except Exception:
                    try:
                        dur = getattr(f, "getFileDuration", lambda: None)()
                        if dur:
                            fs = max(1, int(round(use_samples / float(dur))))
                        else:
                            fs = 250
                    except Exception:
                        fs = 250
                data = np.zeros((use_samples, n_signals), dtype=float)
                for ch in range(n_signals):
                    sig = f.readSignal(ch)
                    if sig is None:
                        sig = np.zeros(use_samples, dtype=float)
                    if len(sig) >= use_samples:
                        sig_use = np.asarray(sig[:use_samples], dtype=float)
                    else:
                        sig_use = np.empty(use_samples, dtype=float)
                        sig_use[:len(sig)] = sig
                        sig_use[len(sig):] = np.nan
                    data[:, ch] = sig_use
                cols = [f"signal_{i + 1}" for i in range(n_signals)]
                df = pd.DataFrame(data, columns=cols)
                df.insert(0, "time", np.arange(use_samples) / float(fs))
                header = {
                    "sampling_frequency": fs,
                    "num_signals": n_signals,
                    "record_name": os.path.basename(edf_path),
                    "num_samples": use_samples
                }
                return df, header
            finally:
                try:
                    f.close()
                except Exception:
                    try:
                        f._close()
                    except Exception:
                        pass
                try:
                    del f
                except Exception:
                    pass
        except Exception as e:
            last_exc = e
            msg = str(e).lower()
            if ("already been opened" in msg) or ("file has already been opened" in msg) or (
                    "resource temporarily unavailable" in msg) or ("i/o error" in msg):
                time.sleep(backoff)
                backoff = min(0.5, backoff * 1.8)
                continue
            time.sleep(backoff)
            backoff = min(0.5, backoff * 1.8)
            continue
    print(f"[read_edf_file] Error reading {edf_path}: {last_exc}")
    return None, None


def apply_signal_filtering(signal, fs, signal_type="ECG"):
    """Bandpass filter for ECG/EEG (simple, zero-phase)."""
    if signal is None or len(signal) < 3:
        return signal
    if signal_type == "ECG":
        low_cutoff, high_cutoff = 0.5, 40.0
    else:  # EEG
        low_cutoff, high_cutoff = 0.5, 70.0
    nyq = fs / 2.0
    low = max(low_cutoff / nyq, 1e-6)
    high = min(high_cutoff / nyq, 0.9999)
    try:
        b, a = butter(4, [low, high], btype='band')
        filtered = filtfilt(b, a, signal)
        return filtered
    except Exception:
        return signal


# ---------- Load patients ----------
def load_patient_data(data_dir, dataset_type="ECG", max_samples=None, max_patients=None,
                      max_records_per_patient=5):
    """
    Load and concatenate all data files found in directory tree.
    Groups files by their immediate parent directory and concatenates vertically.

    Args:
        data_dir: Root directory to search
        dataset_type: "ECG" or "EEG"
        max_samples: Max samples to read per file (None for all)
        max_patients: Maximum number of patient groups to load
        max_records_per_patient: Maximum files to concatenate per group

    Returns:
        List of patient records with concatenated data
    """
    patients = []

    if dataset_type == "ECG":
        file_ext = '.dat'
        concat_func = concatenate_ecg_files
    else:  # EEG
        if not PYEDFLIB_AVAILABLE:
            print("pyedflib is required to read EDF files. Please install: pip install pyedflib")
            return []
        file_ext = '.edf'
        concat_func = concatenate_eeg_files

    # Find all data files grouped by parent directory
    grouped_files = find_all_data_files(data_dir, file_ext)

    if not grouped_files:
        print(f"[load_patient_data] No {dataset_type} files found in {data_dir}")
        return []

    print(f"[load_patient_data] Found {len(grouped_files)} groups of {dataset_type} files")

    # Process each group
    for group_idx, (group_name, file_paths) in enumerate(sorted(grouped_files.items())):
        if max_patients is not None and group_idx >= max_patients:
            break

        print(f"[load_patient_data] Processing group '{group_name}' with {len(file_paths)} files...")

        # Limit files per group if specified
        if max_records_per_patient:
            file_paths = file_paths[:max_records_per_patient]

        # Concatenate all files in this group
        combined_df, combined_header = concat_func(file_paths, max_samples=max_samples)

        if combined_df is None:
            print(f"[load_patient_data] Failed to load group '{group_name}'")
            continue

        # Apply filtering to all signal columns
        fs = combined_header.get("sampling_frequency", 250)
        signal_cols = [c for c in combined_df.columns if c.startswith("signal_")]

        for col in signal_cols:
            combined_df[col] = apply_signal_filtering(combined_df[col].values, fs, dataset_type)

        # Create patient record
        patient_record = {
            "name": group_name,
            "header": combined_header,
            "ecg": combined_df,  # Note: still called 'ecg' even for EEG for compatibility
            "type": dataset_type,
            "source_directory": os.path.dirname(file_paths[0]),
            "files_concatenated": len(file_paths),
            "total_samples": combined_header.get('num_samples', len(combined_df)),
            "total_channels": len(signal_cols)
        }

        patients.append(patient_record)
        print(f"[load_patient_data] Loaded '{group_name}': {len(file_paths)} files, "
              f"{len(combined_df)} samples, {len(signal_cols)} channels")

    print(f"[load_patient_data] Successfully loaded {len(patients)} patient groups total")
    return patients

# ---------- Feature functions ----------
def extract_ecg_features(ecg_df, fs=250):
    features = {}
    if ecg_df is None:
        return features
    for col in ecg_df.columns:
        if not col.startswith("signal_"):
            continue
        sig = ecg_df[col].values
        try:
            height = np.quantile(sig, 0.85)
            peaks, _ = find_peaks(sig, height=height, distance=int(0.3 * fs))
            if peaks.size > 1:
                rr = np.diff(peaks) / fs
                features[col] = {"peaks": peaks.tolist(), "rr": rr.tolist()}
            else:
                features[col] = {"peaks": peaks.tolist(), "rr": []}
        except Exception:
            features[col] = {"peaks": [], "rr": []}
    return features


def extract_eeg_features(eeg_df, fs=250):
    features = {}
    if eeg_df is None:
        return features
    try:
        from scipy import signal as sp_signal
    except Exception:
        return features
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 70)
    }
    for col in eeg_df.columns:
        if not col.startswith("signal_"):
            continue
        sig = eeg_df[col].values
        try:
            freqs, psd = sp_signal.welch(sig, fs, nperseg=min(1024, max(256, len(sig) // 4)))
            band_pows = {}
            for band, (lo, hi) in bands.items():
                mask = (freqs >= lo) & (freqs <= hi)
                band_pows[band] = float(np.trapz(psd[mask], freqs[mask])) if np.any(mask) else 0.0
            total = sum(band_pows.values()) if sum(band_pows.values()) > 0 else 1.0
            rel = {f"{k}_rel": v / total for k, v in band_pows.items()}
            features[col] = {**band_pows, **rel}
        except Exception:
            features[col] = {b: 0.0 for b in bands.keys()}
    return features


# ---------- Buffers & app ----------
def ensure_buffers_for_patient(pid, fs, display_window, rr_capacity=300):
    bufs = GLOBAL_DATA["buffers"]
    if pid not in bufs:
        blen = max(1, int(round(display_window * fs)))
        bufs[pid] = {
            "signal_buffer": np.full(blen, np.nan),
            "write_idx": 0,
            "len": blen,
            "rr_buffer": np.full(rr_capacity, np.nan),
            "rr_write_idx": 0,
            "last_peak_global_index": -1,
            "direction": 1,
            "ping_position": 0.0,
            "ai_analysis": None  # Store AI analysis results
        }
    else:
        bufinfo = bufs[pid]
        desired_len = max(1, int(round(display_window * fs)))
        if bufinfo["len"] != desired_len:
            bufinfo["signal_buffer"] = np.full(desired_len, np.nan)
            bufinfo["len"] = desired_len
            bufinfo["write_idx"] = 0


app = Dash(__name__)
server = app.server
GLOBAL_DATA = {"patients": [], "buffers": {}, "dataset_type": "ECG"}

def capture_graph_screenshot(figure):
    """Capture a screenshot of the current plotly figure and return as bytes."""
    if not SCREENSHOT_AVAILABLE:
        return None
    try:
        img_bytes = pio.to_image(figure, format="png", width=800, height=600, engine="kaleido")
        return img_bytes
    except Exception as e:
        print(f"[Screenshot] Error: {e}")
        return None
# ---------- Layout (3-channel focused) ----------
app.layout = html.Div([
    html.Div([html.H1("Enhanced ECG/EEG Real-time Monitor with AI Analysis (3-Channel Display)",
                      style={'textAlign': 'center', 'color': '#00ff41'}), ],
             style={'backgroundColor': '#000', 'padding': '8px', 'marginBottom': '8px'}),

    html.Div([
        html.Label("Signal Type:"),
        dcc.RadioItems(id="dataset-type", options=[{"label": "ECG", "value": "ECG"}, {"label": "EEG", "value": "EEG"}],
                       value="ECG", labelStyle={'display': 'inline-block', 'marginRight': '12px'}),
        html.Label("Data Directory (optional):", style={'marginLeft': '20px'}),
        dcc.Input(id="data-dir", type="text", value="", style={"width": "50%"}),
        html.Button("Load Data", id="load-btn", n_clicks=0, style={"marginLeft": "12px"}),
        html.Div(id="load-output", style={"marginTop": "6px", "whiteSpace": "pre-wrap"})
    ], style={"padding": "6px", "backgroundColor": "#111", "borderRadius": "6px"}),

    html.Div([
        html.Label("Visualization:"),
        dcc.Dropdown(id="viz-type", options=[
            {"label": "Standard Cyclical Monitor", "value": "icu"},
            {"label": "Ping-Pong Display", "value": "pingpong"},
            {"label": "Polar RR Intervals", "value": "polar"},
            {"label": "Cross Recurrence Plot", "value": "crossrec"}
        ], value="icu", style={"width": "300px"}),

        html.Label("Select Patients:"),
        dcc.Dropdown(id="patients-dropdown", multi=True, style={"width": "400px"}),

        dcc.Checklist(id="collapse", options=[{"label": "Overlay multiple", "value": "overlay"}], value=["overlay"],
                      style={"marginTop": "8px"}),

        # Channel display controls
        html.Div([
            dcc.Checklist(id="show-all-channels",
                          options=[{"label": "Show all available channels", "value": "show_all"}],
                          value=[],
                          style={"marginTop": "8px"}),
            html.Div(id="channel-info", style={"marginTop": "8px", "fontSize": "12px", "color": "#ccc"})
        ])
    ], style={"padding": "8px", "backgroundColor": "#111", "borderRadius": "6px", "marginTop": "8px"}),

    html.Div([
        html.Label("Speed:"),
        dcc.Slider(id="speed", min=0.1, max=10, step=0.1, value=1,
                   marks={0.5: "0.5x", 1: "1x", 2: "2x", 5: "5x", 10: "10x"}),
        html.Label("Update interval (ms):"),
        dcc.Input(id="chunk-ms", type="number", value=200, min=20, step=10),
        html.Label("Window (s):"),
        dcc.Input(id="display-window", type="number", value=8, min=1, step=1),
        html.Button("Play", id="play-btn", n_clicks=0, style={"marginLeft": "8px"}),
        html.Button("Pause", id="pause-btn", n_clicks=0, style={"marginLeft": "8px"}),
        html.Button("Reset", id="reset-btn", n_clicks=0, style={"marginLeft": "8px"}),
    ], style={"padding": "8px", "backgroundColor": "#111", "borderRadius": "6px", "marginTop": "8px"}),

    # AI Analysis Panel
    html.Div([
        html.H3("RNN-Based ECG Analysis", style={'color': '#00ff41'}),
        html.P("Analyzes current ECG signal data using trained RNN model",
               style={"color": "#ccc", "fontSize": "12px", "marginBottom": "10px"}),
        html.Div(id="ai-analysis-output", style={"color": "#fff", "fontSize": "14px", "minHeight": "100px"}),
        html.Button("Run AI Analysis", id="ai-analyze-btn", n_clicks=0,
                    style={"marginTop": "8px", "backgroundColor": "#007acc", "color": "white",
                           "border": "none", "padding": "10px 20px", "borderRadius": "6px",
                           "fontSize": "14px", "fontWeight": "bold", "cursor": "pointer"})
    ], style={"padding": "12px", "backgroundColor": "#111", "borderRadius": "6px", "marginTop": "8px"}),

    html.Hr(),

    dcc.Interval(id="interval", interval=200, n_intervals=0),
    dcc.Store(id="app-state", data=None),

    dcc.Graph(id="main-graph", config={"displayModeBar": True}, style={"height": "520px"}),
    dcc.Graph(id="extra-graph", config={"displayModeBar": True}, style={"height": "320px"}),
], style={"padding": "12px", "backgroundColor": "#222", "color": "#fff", "minHeight": "100vh"})


# ---------- Callbacks (updated for 3-channel display) ----------
@app.callback(
    [Output("load-output", "children"),
     Output("patients-dropdown", "options"),
     Output("channel-info", "children")],
    [Input("load-btn", "n_clicks"),
     Input("dataset-type", "value")],
    [State("data-dir", "value")],
    prevent_initial_call=True
)
def load_data(nc, dataset_type, data_dir):
    if dataset_type == "EEG" and not PYEDFLIB_AVAILABLE:
        return "pyedflib not installed. Install with: pip install pyedflib", [], ""
    if not data_dir or data_dir.strip() == "":
        auto = find_dataset_directory(dataset_type, ".")
        if auto is None:
            return f"No {dataset_type} data found automatically. Please provide directory.", [], ""
        data_dir = auto
    if not os.path.isdir(data_dir):
        return f"Directory not found: {data_dir}", [], ""
    GLOBAL_DATA["dataset_type"] = dataset_type
    if dataset_type == "EEG":
        patients = load_patient_data(data_dir, dataset_type, max_samples=None, max_patients=MAX_EEG_SUBJECTS)
    else:
        patients = load_patient_data(data_dir, dataset_type, max_samples=None, max_patients=None)
    if not patients:
        return f"No {dataset_type} patients found in {data_dir}.", [], ""
    GLOBAL_DATA["patients"] = patients
    GLOBAL_DATA["buffers"] = {}
    for idx, p in enumerate(patients):
        try:
            fs = float(p["header"].get("sampling_frequency", 250.0)) if p.get("header") else 250.0
            ensure_buffers_for_patient(idx, fs, display_window=8)
        except Exception as e:
            print("Buffer init error", e)
    options = [{"label": f"{p['name']} ({p['type']})", "value": idx} for idx, p in enumerate(patients)]

    # Generate channel info
    channel_info = ""
    if patients:
        first_patient = patients[0]
        available_channels = [c for c in first_patient["ecg"].columns if c.startswith("signal_")]
        if dataset_type == "ECG":
            if len(available_channels) == 1:
                channel_info = f"ECG: 1 channel available - will display with copy and inverted versions"
            elif len(available_channels) == 2:
                channel_info = f"ECG: 2 channels available - third channel will be derived (Lead2 - Lead1)"
            elif len(available_channels) >= 3:
                channel_info = f"ECG: {len(available_channels)} channels available - showing main 3 by default"
        else:
            if len(available_channels) <= 3:
                channel_info = f"EEG: {len(available_channels)} channels available - all will be displayed"
            else:
                channel_info = f"EEG: {len(available_channels)} channels available - showing main 3 by default"

    if dataset_type == "EEG":
        msg = f"Loaded {len(patients)} EEG subjects from {data_dir} (limited to {MAX_EEG_SUBJECTS} unique subjects)."
    else:
        msg = f"Loaded {len(patients)} ECG records from {data_dir}."
    return msg, options, channel_info


@app.callback(
    Output("interval", "interval"),
    Input("chunk-ms", "value"),
    Input("speed", "value")
)
def adjust_interval(chunk_ms, speed):
    try:
        cm = max(20, int(float(chunk_ms)))
    except:
        cm = 200
    return cm


# ---------- AI Analysis Callback (updated for 3-channel) ----------
@app.callback(
    Output("ai-analysis-output", "children"),
    Input("ai-analyze-btn", "n_clicks"),
    [State("patients-dropdown", "value"),
     State("show-all-channels", "value"),
     State("app-state", "data")],
    prevent_initial_call=True
)
def run_rnn_ai_analysis(n_clicks, selected_patients, show_all_channels, app_state):
    """Run RNN-based AI analysis on current patient ECG data"""
    try:
        if not n_clicks:
            return "Click 'Run AI Analysis' to analyze current ECG signals using RNN model"

        # Check if we have patients loaded
        patients = GLOBAL_DATA.get("patients", [])
        if not patients:
            return html.Div([
                html.P("No patients loaded. Please load ECG data first.",
                       style={"color": "#ff6347"})
            ])

        # Get selected patient indices
        selected_idxs = []
        if selected_patients:
            try:
                if isinstance(selected_patients, list):
                    selected_idxs = [int(i) for i in selected_patients if 0 <= int(i) < len(patients)]
                else:
                    idx = int(selected_patients)
                    if 0 <= idx < len(patients):
                        selected_idxs = [idx]
            except:
                selected_idxs = []

        if not selected_idxs:
            selected_idxs = [0]  # Default to first patient

        # Try to load the RNN model
        try:
            model_loaded = AI_MODEL.load_model()
            if not model_loaded:
                return html.Div([
                    html.P("Failed to load RNN model.",
                           style={"color": "#ff6347"}),
                    html.P("Please ensure these files exist in the script directory:",
                           style={"color": "#ccc", "fontSize": "12px"}),
                    html.Ul([
                        html.Li("ECG.weights.h5 (your trained RNN weights)"),
                        html.Li("labels.txt (class labels, one per line)")
                    ], style={"color": "#ffd700", "fontSize": "12px"}),
                    html.P("Note: TensorFlow and Keras must be installed for RNN functionality.",
                           style={"color": "#ccc", "fontSize": "11px"})
                ])
        except Exception as model_error:
            return html.Div([
                html.P(f"Error loading RNN model: {str(model_error)}",
                       style={"color": "#ff6347"}),
                html.P("Check that TensorFlow/Keras is properly installed: pip install tensorflow",
                       style={"color": "#ccc", "fontSize": "12px"})
            ])

        # Analyze each selected patient
        all_results = []
        dataset_type = GLOBAL_DATA.get("dataset_type", "ECG")
        show_all = "show_all" in (show_all_channels or [])

        for pid in selected_idxs[:3]:  # Limit to 3 patients to avoid UI clutter
            if pid >= len(patients):
                continue

            patient = patients[pid].copy()
            patient_name = patient.get("name", f"Patient {pid}")

            # Check if patient has ECG data
            if "ecg" not in patient or patient["ecg"] is None:
                all_results.append(html.Div([
                    html.H4(f"{patient_name}:", style={"color": "#00ff41"}),
                    html.P("No ECG data available for analysis", style={"color": "#ff6347"})
                ]))
                continue

            # Get current playback position
            pos = 0
            if app_state and "pos" in app_state and pid < len(app_state["pos"]):
                pos = app_state["pos"][pid]

            if pos <= 0:
                all_results.append(html.Div([
                    html.H4(f"{patient_name}:", style={"color": "#00ff41"}),
                    html.P("No signal data played yet. Start playback first.",
                           style={"color": "#ff6347"})
                ]))
                continue

            # Process patient channels
            channel_result = process_patient_channels(patient, dataset_type, show_all)
            channels_to_analyze = channel_result["channels"]

            if not channels_to_analyze:
                all_results.append(html.Div([
                    html.H4(f"{patient_name}:", style={"color": "#00ff41"}),
                    html.P("No channels available for analysis", style={"color": "#ff6347"})
                ]))
                continue

            # Get signal data up to current position
            ecg_data = patient["ecg"].iloc[:pos].copy()

            if len(ecg_data) < 100:  # Need minimum amount of data
                all_results.append(html.Div([
                    html.H4(f"{patient_name}:", style={"color": "#00ff41"}),
                    html.P(f"Insufficient data for analysis ({len(ecg_data)} samples). Need at least 100 samples.",
                           style={"color": "#ff6347"})
                ]))
                continue

            # Analyze primary channel (first available)
            primary_channel = channels_to_analyze[0]

            try:
                print(f"[RNN Analysis] Analyzing {patient_name}, channel: {primary_channel}")

                # Run RNN analysis
                analysis_start_time = time.time()
                analysis_result = AI_MODEL.analyze_patient_data(
                    ecg_data,
                    channel_name=primary_channel,
                    top_k=5
                )
                analysis_duration = time.time() - analysis_start_time

                if "error" in analysis_result:
                    all_results.append(html.Div([
                        html.H4(f"{patient_name}:", style={"color": "#00ff41"}),
                        html.P(f"Analysis Error: {analysis_result['error']}",
                               style={"color": "#ff6347"})
                    ]))
                    continue

                # Format results for display
                predictions = analysis_result.get("predictions", [])
                if not predictions:
                    all_results.append(html.Div([
                        html.H4(f"{patient_name}:", style={"color": "#00ff41"}),
                        html.P("No predictions returned from RNN model.",
                               style={"color": "#ff6347"})
                    ]))
                    continue

                # Create result elements for this patient
                patient_elements = []

                # Patient header
                patient_elements.append(
                    html.H4(f"{patient_name}:", style={"color": "#00ff41", "marginBottom": "8px"})
                )

                # Analysis info
                signal_length = analysis_result.get("signal_length", len(ecg_data))
                inference_time = analysis_result.get("inference_time_s", 0)
                patient_elements.append(
                    html.P(f"Analyzed {signal_length} samples in {analysis_duration:.2f}s (RNN: {inference_time:.3f}s)",
                           style={"color": "#ccc", "fontSize": "11px", "fontStyle": "italic"})
                )

                # Top predictions
                for i, pred in enumerate(predictions[:3]):  # Show top 3
                    confidence = pred.get("confidence", 0)
                    confidence_pct = f"{confidence * 100:.1f}%"
                    label = pred.get("label", "Unknown")

                    # Color coding
                    if confidence >= 0.8:
                        confidence_color = "#00ff41"  # Green
                        icon = "✓"
                    elif confidence >= 0.6:
                        confidence_color = "#ffd700"  # Yellow
                        icon = "?"
                    else:
                        confidence_color = "#ff6347"  # Red
                        icon = "!"

                    patient_elements.append(
                        html.Div([
                            html.Span(f"{i + 1}. ", style={"fontWeight": "bold", "color": "#ccc"}),
                            html.Span(f"{icon} ", style={"marginRight": "5px", "color": confidence_color}),
                            html.Span(f"{label}", style={"color": "#00bfff", "fontWeight": "bold"}),
                            html.Span(f" ({confidence_pct})", style={"color": confidence_color, "marginLeft": "10px"})
                        ], style={
                            "margin": "2px 0",
                            "padding": "6px",
                            "backgroundColor": "#333",
                            "borderRadius": "3px",
                            "borderLeft": f"3px solid {confidence_color}"
                        })
                    )

                # Channel info
                patient_elements.append(
                    html.P(f"Channel: {primary_channel} | {channel_result['derived_info']}",
                           style={"fontSize": "11px", "color": "#888", "fontStyle": "italic", "marginTop": "8px"})
                )

                # Add to all results
                all_results.append(
                    html.Div(patient_elements, style={"marginBottom": "20px"})
                )

            except Exception as analysis_error:
                all_results.append(html.Div([
                    html.H4(f"{patient_name}:", style={"color": "#00ff41"}),
                    html.P(f"Analysis failed: {str(analysis_error)}",
                           style={"color": "#ff6347"})
                ]))
                print(f"[RNN Analysis] Error analyzing {patient_name}: {analysis_error}")

        # Add model info footer
        if all_results:
            model_info = AI_MODEL.get_model_info()
            all_results.append(html.Hr(style={"borderColor": "#555", "margin": "15px 0 10px 0"}))
            all_results.append(
                html.P([
                    html.Span("RNN Model Info: ", style={"color": "#ccc", "fontSize": "11px"}),
                    html.Span(f"{model_info.get('num_classes', 0)} classes, ",
                              style={"color": "#00bfff", "fontSize": "11px"}),
                    html.Span(f"sequence length: {model_info.get('sequence_length', 0)}, ",
                              style={"color": "#00bfff", "fontSize": "11px"}),
                    html.Span(f"predictions made: {model_info.get('total_predictions', 0)}",
                              style={"color": "#00bfff", "fontSize": "11px"})
                ])
            )

            # Timestamp
            all_results.append(
                html.P(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                       style={"color": "#888", "fontSize": "10px", "marginTop": "5px"})
            )

        if not all_results:
            return "No analysis results available. Ensure patients are loaded and have played some data."

        return html.Div(all_results, style={"lineHeight": "1.4"})

    except Exception as e:
        # Catch-all error handler
        print(f"[RNN Analysis] Unexpected error: {e}")
        return html.Div([
            html.P("Unexpected error in RNN analysis.",
                   style={"color": "#ff6347"}),
            html.P(f"Error details: {str(e)}",
                   style={"color": "#ccc", "fontSize": "11px"}),
            html.P("Check the console for more information.",
                   style={"color": "#ccc", "fontSize": "12px"})
        ])

# ---------- Helper to build each viz figure (updated for 3-channel display) ----------
def make_viz_figure(viz_type, patients, selected_idxs, show_all_channels, state,
                    display_window_val=8.0, speed_val=1.0, collapse_flag=None):
    """Return a plotly Figure for the requested viz_type with 3-channel focus."""
    try:
        if not patients:
            return create_empty_figure("No patients")
        dataset_type = GLOBAL_DATA.get("dataset_type", "ECG")
        show_all = "show_all" in (show_all_channels or [])

        # ICU Monitor - 3 channels focus
        if viz_type == "icu":
            fig = go.Figure()
            color_map = ["#00ff41", "#00bfff", "#ffd700", "#ff6347", "#da70d6", "#7fffd4", "#ff7f50", "#7fff00"]
            color_idx = 0

            for pid in selected_idxs:
                if pid >= len(patients):
                    continue

                # Work with a copy and process channels
                p = patients[pid].copy()
                if "ecg" not in p or p["ecg"] is None:
                    continue

                # Process patient channels
                channel_result = process_patient_channels(p, dataset_type, show_all)
                channels_to_display = channel_result["channels"]

                if not channels_to_display:
                    continue

                fs = float(p["header"].get("sampling_frequency", 250.0)) if p.get("header") else 250.0
                pos = state["pos"][pid] if state and "pos" in state and pid < len(state["pos"]) else 0
                win = int(display_window_val * fs)
                start = max(0, pos - win)

                # Display channels (limit to 3 for ICU view unless show_all is True)
                display_channels = channels_to_display if show_all else channels_to_display[:3]

                for ch in display_channels:
                    if ch not in p["ecg"].columns:
                        continue
                    seg = p["ecg"].iloc[start:pos][["time", ch]]
                    if seg.shape[0] == 0:
                        continue
                    times = (seg["time"].values - seg["time"].values[0]).astype(float)
                    vals = seg[ch].values
                    color = color_map[color_idx % len(color_map)]

                    # Vertical offset for non-overlay mode
                    yoff = 0 if ("overlay" in (collapse_flag or [])) else color_idx * (np.nanstd(vals) * 3 + 0.1)

                    # Channel name for legend
                    if ch.startswith("derived_"):
                        ch_name = f"{p['name']} Derived({ch.split('_')[-2]}-{ch.split('_')[-1]})"
                    elif ch.endswith("_copy"):
                        ch_name = f"{p['name']} {ch.replace('_copy', ' Copy')}"
                    elif ch.endswith("_inverted"):
                        ch_name = f"{p['name']} {ch.replace('_inverted', ' Inverted')}"
                    else:
                        ch_name = f"{p['name']} {ch}"

                    fig.add_trace(go.Scattergl(x=times, y=vals + yoff, mode="lines",
                                               line=dict(color=color, width=2),
                                               name=ch_name))
                    color_idx += 1

            fig.update_layout(template="plotly_dark",
                              title=f"{dataset_type} Real-time Monitor (3-Channel Focus)",
                              xaxis=dict(title="Time (s)"),
                              yaxis=dict(title="Amplitude"))
            return fig

        # Ping-pong: use first selected patient, 3 channels
        if viz_type == "pingpong":
            if not selected_idxs:
                return create_empty_figure("No patients selected")
            pid = selected_idxs[0]
            if pid >= len(patients):
                return create_empty_figure("Invalid patient")

            p = patients[pid].copy()
            if "ecg" not in p or p["ecg"] is None:
                return create_empty_figure("No data for selected patient")

            # Process channels
            channel_result = process_patient_channels(p, dataset_type, show_all)
            channels_to_display = channel_result["channels"]

            if not channels_to_display:
                return create_empty_figure("No channels available")

            fs = float(p["header"].get("sampling_frequency", 250.0)) if p.get("header") else 250.0
            buf = GLOBAL_DATA["buffers"].get(pid, {})
            if "ping_position" not in buf:
                buf["ping_position"] = 0.0
                buf["direction"] = 1

            if state and state.get("playing", False):
                step = 0.01 * speed_val
                buf["ping_position"] += buf["direction"] * step
                if buf["ping_position"] >= 1.0:
                    buf["ping_position"] = 1.0
                    buf["direction"] = -1
                if buf["ping_position"] <= 0.0:
                    buf["ping_position"] = 0.0
                    buf["direction"] = 1

            pos = state["pos"][pid] if state and "pos" in state and pid < len(state["pos"]) else 0
            total_avail = min(pos, len(p["ecg"]))
            win = int(display_window_val * fs)
            if total_avail < win:
                start = 0
                end = total_avail
            else:
                max_start = total_avail - win
                start = int(buf["ping_position"] * max_start)
                end = start + win

            fig = go.Figure()
            ch = channels_to_display[0] if channels_to_display else "signal_1"
            if ch in p["ecg"].columns:
                seg = p["ecg"].iloc[start:end][["time", ch]]
                if seg.shape[0] > 0:
                    times = (seg["time"].values - seg["time"].values[0]).astype(float)
                    vals = seg[ch].values
                    fig.add_trace(go.Scattergl(x=times, y=vals, mode="lines",
                                               line=dict(color="#00ffff", width=2),
                                               name=f"{p['name']} {ch}"))
                    fig.update_layout(template="plotly_dark",
                                      title=f"Ping-Pong: {p['name']} ({ch})",
                                      xaxis=dict(title="Time (s)"),
                                      yaxis=dict(title="Amplitude"))
                    return fig
            return create_empty_figure("No channel data")

        # Polar - 3 channels focus
        if viz_type == "polar":
            fig = go.Figure()
            palette = ["#ff7f50", "#7fff00", "#1e90ff", "#da70d6", "#ffd700", "#00fa9a"]

            if dataset_type == "ECG":
                for k, pid in enumerate(selected_idxs):
                    if pid >= len(patients):
                        continue
                    buf = GLOBAL_DATA["buffers"].get(pid)
                    if not buf:
                        continue
                    rr = buf["rr_buffer"]
                    wi = int(buf["rr_write_idx"])
                    ordered = np.concatenate((rr[wi:], rr[:wi]))
                    ordered = ordered[~np.isnan(ordered)]
                    if ordered.size == 0:
                        continue
                    n = ordered.size
                    angles = np.linspace(0, 360, n, endpoint=False)
                    mn, mx = ordered.min(), ordered.max()
                    if mx - mn < 1e-6:
                        rnorm = np.ones_like(ordered) * 0.5
                    else:
                        rnorm = 0.2 + 0.8 * (ordered - mn) / (mx - mn)
                    fig.add_trace(go.Scatterpolar(r=rnorm, theta=angles, mode="lines+markers",
                                                  name=patients[pid]["name"],
                                                  line=dict(color=palette[k % len(palette)])))
                fig.update_layout(template="plotly_dark", title="Polar RR Intervals",
                                  polar=dict(radialaxis=dict(visible=True)))
                return fig
            else:
                for k, pid in enumerate(selected_idxs):
                    p = patients[pid].copy()
                    if "features" not in p:
                        p["features"] = extract_eeg_features(p["ecg"], p["header"].get("sampling_frequency", 250))

                    # Process channels and use first available
                    channel_result = process_patient_channels(p, dataset_type, show_all)
                    channels_to_display = channel_result["channels"]

                    ch = channels_to_display[0] if channels_to_display else next(
                        (c for c in p["ecg"].columns if c.startswith("signal_")), None)
                    if not ch or ch not in p["features"]:
                        continue
                    feats = p["features"][ch]
                    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
                    vals = [feats.get(b, 0.0) for b in bands]
                    if sum(vals) <= 0:
                        continue
                    mx = max(vals)
                    rnorm = [0.2 + 0.8 * (v / mx) if mx > 0 else 0.2 for v in vals]
                    angles = [0, 72, 144, 216, 288]
                    fig.add_trace(go.Scatterpolar(r=rnorm, theta=angles, mode="lines+markers",
                                                  name=f"{p['name']}:{ch}",
                                                  line=dict(color=palette[k % len(palette)])))
                fig.update_layout(template="plotly_dark", title="EEG Band Powers (Polar)",
                                  polar=dict(radialaxis=dict(visible=True)))
                return fig

        # Cross recurrence - keep original logic for even channel comparison
        if viz_type == "crossrec":
            if not selected_idxs:
                return create_empty_figure("No patient selected")
            pid = selected_idxs[0]
            if pid >= len(patients):
                return create_empty_figure("Invalid patient")

            p = patients[pid].copy()
            if "ecg" not in p or p["ecg"] is None:
                return create_empty_figure("No data for patient")

            ecg = p["ecg"]
            pos = max(1, state["pos"][pid]) if state and "pos" in state and pid < len(state["pos"]) else 1
            nmax = min(pos, 500)
            if nmax < 2:
                return create_empty_figure("Not enough data yet")

            # Process channels
            channel_result = process_patient_channels(p, dataset_type, show_all)
            available_channels = channel_result["channels"]

            if dataset_type == "ECG":
                if len(available_channels) < 2:
                    ch1 = available_channels[0]
                    s1 = ecg[ch1].values[pos - nmax:pos]
                    s2 = ecg[ch1].values[pos - nmax + 10: pos + 10] if pos + 10 <= len(ecg) else s1
                    ch2 = f"{ch1} (delayed)"
                else:
                    ch1, ch2 = available_channels[0], available_channels[1]
                    s1 = ecg[ch1].values[pos - nmax:pos]
                    s2 = ecg[ch2].values[pos - nmax:pos]
            else:
                if len(available_channels) < 2 or (len(available_channels) % 2) != 0:
                    return create_empty_figure(
                        "For EEG cross-recurrence, need an even number of channels (e.g. 2,4,6)")
                half = len(available_channels) // 2
                group1 = available_channels[:half]
                group2 = available_channels[half:]
                s1 = np.mean(ecg[group1].values[pos - nmax:pos], axis=1)
                s2 = np.mean(ecg[group2].values[pos - nmax:pos], axis=1)
                ch1 = "+".join(group1)
                ch2 = "+".join(group2)

            try:
                thresh = 0.12 * max(np.std(s1), np.std(s2), 1e-6)
                rec = (np.abs(s1[:, None] - s2[None, :]) < thresh).astype(int)
                hm = go.Heatmap(z=rec, colorscale=[[0, '#000'], [1, '#00ffbf']], showscale=True)
                fig = go.Figure(data=[hm])
                fig.update_layout(template="plotly_dark", title=f"Cross Recurrence: {p['name']}",
                                  xaxis=dict(title=""), yaxis=dict(title=""))
                return fig
            except Exception as e:
                print("Crossrec error:", e)
                return create_empty_figure("Cross-recurrence computation failed")

    except Exception as e:
        print("make_viz_figure error:", e)
        return create_empty_figure("Error building viz")


def create_empty_figure(title="No data"):
    fig = go.Figure()
    fig.update_layout(template="plotly_dark", title={"text": title, "font": {"color": "#fff"}})
    return fig


# ---------- Combined update callback (updated for 3-channel display) ----------
@app.callback(
    [Output("main-graph", "figure"),
     Output("extra-graph", "figure"),
     Output("app-state", "data")],
    [Input("interval", "n_intervals"),
     Input("play-btn", "n_clicks"),
     Input("pause-btn", "n_clicks"),
     Input("reset-btn", "n_clicks")],
    [State("app-state", "data"),
     State("patients-dropdown", "value"),
     State("show-all-channels", "value"),
     State("viz-type", "value"),
     State("speed", "value"),
     State("chunk-ms", "value"),
     State("display-window", "value"),
     State("collapse", "value")],
    prevent_initial_call=False
)
def combined_update(n_intervals, n_play, n_pause, n_reset, state,
                    selected, show_all_channels, viz_type, speed, chunk_ms, display_window, collapse_flag):
    try:
        patients = GLOBAL_DATA.get("patients", [])
        if not patients:
            return create_empty_figure("No patients loaded"), create_empty_figure(), state or {"playing": False,
                                                                                               "pos": [],
                                                                                               "write_idx": []}
        if state is None:
            state = {"playing": False, "pos": [0] * len(patients), "write_idx": [0] * len(patients)}
        if "pos" not in state or len(state["pos"]) != len(patients):
            state["pos"] = [0] * len(patients)
        if "write_idx" not in state or len(state["write_idx"]) != len(patients):
            state["write_idx"] = [0] * len(patients)
        if "playing" not in state:
            state["playing"] = False

        ctx = callback_context
        trigger = getattr(ctx, "triggered_id", None)
        if trigger is None:
            if ctx.triggered:
                trigger = ctx.triggered[0]["prop_id"].split(".")[0]

        if trigger == "play-btn":
            state["playing"] = True
        elif trigger == "pause-btn":
            state["playing"] = False
        elif trigger == "reset-btn":
            state["playing"] = False
            state["pos"] = [0] * len(patients)
            state["write_idx"] = [0] * len(patients)
            for pid in list(GLOBAL_DATA.get("buffers", {}).keys()):
                buf = GLOBAL_DATA["buffers"][pid]
                buf["signal_buffer"].fill(np.nan)
                buf["write_idx"] = 0
                buf["rr_buffer"].fill(np.nan)
                buf["rr_write_idx"] = 0
                buf["last_peak_global_index"] = -1
                buf["direction"] = 1
                buf["ping_position"] = 0.0
                buf["ai_analysis"] = None

        try:
            chunk_ms_val = max(20, float(chunk_ms or 200))
        except:
            chunk_ms_val = 200.0
        try:
            speed_val = max(0.1, float(speed or 1.0))
        except:
            speed_val = 1.0
        try:
            display_window_val = max(1.0, float(display_window or 8.0))
        except:
            display_window_val = 8.0

        for pid, p in enumerate(patients):
            try:
                fs = float(p["header"].get("sampling_frequency", 250.0)) if p.get("header") else 250.0
            except:
                fs = 250.0
            ensure_buffers_for_patient(pid, fs, display_window_val)

        if trigger == "interval" and state.get("playing", False):
            for pid, p in enumerate(patients):
                if not p or "ecg" not in p or p["ecg"] is None:
                    continue
                ecg = p["ecg"]
                if ecg.shape[0] == 0:
                    continue
                try:
                    fs = float(p["header"].get("sampling_frequency", 250.0)) if p.get("header") else 250.0
                except:
                    fs = 250.0
                chunk_sec = (chunk_ms_val / 1000.0) * speed_val
                chunk_samples = max(1, int(round(chunk_sec * fs)))
                pos0 = state["pos"][pid]
                pos1 = min(len(ecg), pos0 + chunk_samples)
                if pos1 > pos0 and "signal_1" in ecg.columns:
                    block = ecg["signal_1"].values[pos0:pos1]
                    buf = GLOBAL_DATA["buffers"][pid]
                    N = buf["len"]
                    L = block.size
                    w0 = buf["write_idx"]
                    if L > 0:
                        if L >= N:
                            buf["signal_buffer"][:] = block[-N:]
                            write_idx = 0
                        else:
                            first_len = min(L, N - w0)
                            if first_len > 0:
                                buf["signal_buffer"][w0:w0 + first_len] = block[:first_len]
                            rem = L - first_len
                            if rem > 0:
                                buf["signal_buffer"][:rem] = block[first_len:]
                            write_idx = (w0 + L) % N
                        buf["write_idx"] = int(write_idx)
                        state["write_idx"][pid] = int(write_idx)
                        if GLOBAL_DATA.get("dataset_type", "ECG") == "ECG":
                            try:
                                tailK = min(int(fs * 10), len(ecg))
                                tail_start = max(0, pos1 - tailK)
                                tail = ecg["signal_1"].values[tail_start:pos1]
                                if len(tail) > 10:
                                    height = np.quantile(tail, 0.85)
                                    peaks_tail, _ = find_peaks(tail, height=height, distance=int(0.3 * fs))
                                    peaks_global = peaks_tail + tail_start
                                    lastp = buf.get("last_peak_global_index", -1)
                                    newp = peaks_global[peaks_global > lastp]
                                    if newp.size > 0:
                                        seq = []
                                        if lastp >= 0:
                                            seq.append(lastp)
                                        seq.extend(newp.tolist())
                                        for i in range(1, len(seq)):
                                            rr = (seq[i] - seq[i - 1]) / float(fs)
                                            if 0.25 <= rr <= 3.0:
                                                ri = int(buf["rr_write_idx"])
                                                buf["rr_buffer"][ri % buf["rr_buffer"].size] = rr
                                                buf["rr_write_idx"] = (ri + 1) % buf["rr_buffer"].size
                                        buf["last_peak_global_index"] = int(newp[-1])
                            except Exception:
                                pass
                state["pos"][pid] = pos1
            try:
                if all(state["pos"][i] >= len(patients[i]["ecg"]) for i in range(len(patients))):
                    state["playing"] = False
            except Exception:
                pass

        selected_idxs = []
        if selected:
            try:
                if isinstance(selected, list):
                    selected_idxs = [int(i) for i in selected if 0 <= int(i) < len(patients)]
                else:
                    idx = int(selected)
                    if 0 <= idx < len(patients):
                        selected_idxs = [idx]
            except:
                selected_idxs = []
        if not selected_idxs:
            selected_idxs = list(range(len(patients)))

        # Build current viz (main figure) with 3-channel focus
        main_fig = make_viz_figure(viz_type, patients, selected_idxs, show_all_channels, state,
                                   display_window_val=display_window_val, speed_val=speed_val,
                                   collapse_flag=collapse_flag)
        extra_fig = create_empty_figure()
        return main_fig, extra_fig, state
    except Exception as e:
        print("combined_update error:", e)
        return create_empty_figure("Error occurred"), create_empty_figure(), {"playing": False, "pos": [],
                                                                              "write_idx": []}


# ---------- Run ----------
if __name__ == "__main__":

    if not PYEDFLIB_AVAILABLE:
        print("Warning: pyedflib not installed. EEG (EDF) support will be disabled until it's installed.")
    app.run(debug=True, host="127.0.0.1", port=8052)