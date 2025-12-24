#!/usr/bin/env python3
"""
YAMNet-based sound classifier using Google Coral Edge TPU.

This version uses YAMNet with spectrogram input (yamnet_spectra_in_edgetpu.tflite).
Audio is converted to mel spectrogram before feeding to the model.

YAMNet is trained on AudioSet and can classify 521 different audio events.
It expects 16kHz mono audio input - we resample if needed.

UDP Stream Format:
- Broadcast IP: 255.255.255.255
- Port: 50005
- Format: 16-bit PCM, mono, little endian, 256 samples per packet
"""

import socket
import struct
import numpy as np
import csv
from datetime import datetime
from pathlib import Path
import logging
import os
import wave
import time
from typing import Dict, Optional, List, Tuple
from scipy import signal as scipy_signal
from scipy.io import wavfile

# Check if we're in mock mode (no Coral hardware)
MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"
# Debug mode: save received audio to file for verification
DEBUG_AUDIO = os.getenv("DEBUG_AUDIO", "false").lower() == "true"
DEBUG_AUDIO_PATH = os.getenv("DEBUG_AUDIO_PATH", "/app/debug_audio")

if MOCK_MODE:
    print("Running in MOCK MODE - no Coral hardware required")
    make_interpreter = None
else:
    try:
        from pycoral.utils.edgetpu import make_interpreter
    except ImportError as e:
        print(f"Error importing pycoral: {e}")
        print("Trying fallback to TensorFlow Lite...")
        try:
            try:
                from tflite_runtime.interpreter import Interpreter
            except ImportError:
                import tensorflow as tf
                Interpreter = tf.lite.Interpreter
            
            def make_interpreter(model_path):
                return Interpreter(model_path=model_path)
            
            print("Using TensorFlow Lite (CPU mode, no Coral acceleration)")
        except ImportError:
            print("Set MOCK_MODE=true to test without Coral hardware")
            exit(1)


# Configuration
CONFIG = {
    "UDP_BROADCAST_IP": "255.255.255.255",
    "LISTENERS": [
        {"name": "default", "port": 50005},
    ],
    # Input sample rate from ESP (we'll resample to 16kHz for YAMNet)
    "INPUT_SAMPLE_RATE": int(os.getenv("INPUT_SAMPLE_RATE", "16000")),
    # YAMNet requires 16kHz
    "YAMNET_SAMPLE_RATE": 16000,
    "SAMPLES_PER_PACKET": 256,
    # YAMNet model and labels (spectrogram input version)
    "MODEL_PATH": os.getenv("MODEL_PATH", "yamnet_spectra_edgetpu.tflite"),
    "LABELS_PATH": os.getenv("LABELS_PATH", "yamnet_class_map.csv"),
    # Minimum confidence to consider a detection valid (0-100)
    "CERTAINTY_THRESHOLD": int(os.getenv("CERTAINTY_THRESHOLD", "30")),
    # Minimum gap between top and second prediction
    "MIN_CONFIDENCE_GAP": int(os.getenv("MIN_CONFIDENCE_GAP", "10")),
    # Ambient noise threshold (RMS level 0.0-1.0)
    "AMBIENT_THRESHOLD": float(os.getenv("AMBIENT_THRESHOLD", "0.01")),
    # Number of consecutive consistent predictions required
    "SMOOTHING_WINDOW": int(os.getenv("SMOOTHING_WINDOW", "2")),
    # YAMNet expects 0.96 seconds of audio for spectrogram (96 frames * 10ms hop)
    "AUDIO_BUFFER_SECONDS": float(os.getenv("AUDIO_BUFFER_SECONDS", "0.96")),
    # Classes of interest (subset of YAMNet's 521 classes)
    # Leave empty to report all classes
    "CLASSES_OF_INTEREST": os.getenv("CLASSES_OF_INTEREST", "").split(",") if os.getenv("CLASSES_OF_INTEREST") else [],
}

# Setup logging
log_level = logging.DEBUG if DEBUG_AUDIO else logging.INFO
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MelSpectrogramGenerator:
    """
    Generate mel spectrograms compatible with YAMNet spectrogram-input model.
    
    YAMNet expects:
    - Input: 0.96s of 16kHz audio
    - Spectrogram: 96 frames x 64 mel bands
    - Frame length: 25ms (400 samples)
    - Frame hop: 10ms (160 samples)
    """
    
    # YAMNet spectrogram parameters
    SAMPLE_RATE = 16000
    STFT_WINDOW_SECONDS = 0.025  # 25ms
    STFT_HOP_SECONDS = 0.010  # 10ms
    MEL_BANDS = 64
    MEL_MIN_HZ = 125.0
    MEL_MAX_HZ = 7500.0
    NUM_FRAMES = 96  # Expected by the model
    
    def __init__(self):
        self.window_length = int(self.SAMPLE_RATE * self.STFT_WINDOW_SECONDS)  # 400
        self.hop_length = int(self.SAMPLE_RATE * self.STFT_HOP_SECONDS)  # 160
        self.fft_length = 512  # Standard for 400-sample windows
        self.mel_filterbank = self._create_mel_filterbank()
        logger.info(f"MelSpectrogram: window={self.window_length}, hop={self.hop_length}, "
                   f"fft={self.fft_length}, mel_bands={self.MEL_BANDS}")
        
    def _create_mel_filterbank(self) -> np.ndarray:
        """Create mel filterbank matrix."""
        # Number of FFT bins
        num_fft_bins = self.fft_length // 2 + 1
        
        # Mel scale conversion functions
        def hz_to_mel(hz):
            return 2595.0 * np.log10(1.0 + hz / 700.0)
        
        def mel_to_hz(mel):
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
        
        # Create mel frequency points
        mel_low = hz_to_mel(self.MEL_MIN_HZ)
        mel_high = hz_to_mel(self.MEL_MAX_HZ)
        mel_points = np.linspace(mel_low, mel_high, self.MEL_BANDS + 2)
        hz_points = mel_to_hz(mel_points)
        
        # Convert Hz to FFT bin indices
        fft_freqs = np.linspace(0, self.SAMPLE_RATE / 2, num_fft_bins)
        
        # Create filterbank
        filterbank = np.zeros((self.MEL_BANDS, num_fft_bins))
        
        for i in range(self.MEL_BANDS):
            # Lower and upper frequencies
            lower = hz_points[i]
            center = hz_points[i + 1]
            upper = hz_points[i + 2]
            
            for j, freq in enumerate(fft_freqs):
                if lower <= freq < center:
                    filterbank[i, j] = (freq - lower) / (center - lower)
                elif center <= freq < upper:
                    filterbank[i, j] = (upper - freq) / (upper - center)
                    
        return filterbank.astype(np.float32)
    
    def compute_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute mel spectrogram from audio waveform.
        
        Args:
            audio: Float32 audio samples at 16kHz, range [-1, 1]
            
        Returns:
            Mel spectrogram of shape (96, 64) - log mel energies
        """
        # Ensure we have enough samples for desired frames
        # 96 frames = 96 * 160 (hop) + 400 (window) - 160 = 15600 samples
        required_samples = (self.NUM_FRAMES - 1) * self.hop_length + self.window_length
        
        if len(audio) < required_samples:
            audio = np.pad(audio, (0, required_samples - len(audio)))
        elif len(audio) > required_samples:
            audio = audio[:required_samples]
        
        # Create Hanning window
        window = np.hanning(self.window_length).astype(np.float32)
        
        # Compute STFT frames
        num_frames = self.NUM_FRAMES
        frames = np.zeros((num_frames, self.fft_length), dtype=np.float32)
        
        for i in range(num_frames):
            start = i * self.hop_length
            frame = audio[start:start + self.window_length] * window
            # Zero-pad to FFT length
            frames[i, :len(frame)] = frame
        
        # Compute FFT and power spectrum
        fft_result = np.fft.rfft(frames, axis=1)
        power_spectrum = np.abs(fft_result) ** 2
        
        # Apply mel filterbank
        mel_spectrum = np.dot(power_spectrum, self.mel_filterbank.T)
        
        # Convert to log scale (add small epsilon to avoid log(0))
        log_mel = np.log(mel_spectrum + 1e-10)
        
        # Normalize to roughly match YAMNet's expected input range
        # YAMNet typically expects values in a specific range after log
        log_mel = np.clip(log_mel, -10, 10)
        
        return log_mel.astype(np.float32)


class YAMNetClassifier:
    """YAMNet-based sound classifier using Google Coral Edge TPU (spectrogram input)."""
    
    def __init__(self, model_path: str, labels_path: str):
        """
        Initialize the YAMNet classifier.
        
        Args:
            model_path: Path to yamnet_spectra_edgetpu.tflite
            labels_path: Path to yamnet_class_map.csv
        """
        self.model_path = model_path
        self.labels_path = labels_path
        self.labels = self._load_labels()
        self.interpreter = self._init_interpreter()
        self.spectrogram_gen = MelSpectrogramGenerator()
        self.input_shape = None
        self.input_dtype = None
        self._setup_io()
        logger.info(f"Loaded {len(self.labels)} YAMNet labels")
        
    def _load_labels(self) -> List[str]:
        """Load YAMNet labels from CSV file."""
        labels = []
        try:
            with open(self.labels_path, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header: index,mid,display_name
                for row in reader:
                    if len(row) >= 3:
                        labels.append(row[2])  # display_name
        except FileNotFoundError:
            logger.error(f"Labels file not found: {self.labels_path}")
        except Exception as e:
            logger.error(f"Error loading labels: {e}")
        return labels
    
    def _init_interpreter(self):
        """Initialize TFLite interpreter with Edge TPU support."""
        if MOCK_MODE:
            logger.info("MOCK MODE: Skipping Coral Edge TPU initialization")
            return None
        try:
            interpreter = make_interpreter(self.model_path)
            interpreter.allocate_tensors()
            logger.info("Initialized YAMNet on Coral Edge TPU")
            return interpreter
        except Exception as e:
            logger.error(f"Failed to initialize interpreter: {e}")
            raise
    
    def _setup_io(self):
        """Setup input/output tensor info."""
        if MOCK_MODE:
            # Spectrogram input: (1, 96, 64, 1)
            self.input_shape = (1, 96, 64, 1)
            self.input_dtype = np.float32
            self.uses_spectrogram = True
            return
            
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        
        self.input_shape = tuple(input_details[0]['shape'])
        self.input_dtype = input_details[0]['dtype']
        self.input_index = input_details[0]['index']
        self.output_index = output_details[0]['index']
        
        logger.info(f"YAMNet input shape: {self.input_shape}, dtype: {self.input_dtype}")
        logger.info(f"YAMNet output shape: {output_details[0]['shape']}")
        
        # Determine if model expects spectrogram or raw audio
        # Spectrogram input: (1, 96, 64, 1) or (1, 96, 64)
        # Raw audio input: (1, 15600) or (15600,)
        self.uses_spectrogram = len(self.input_shape) >= 3 or (
            len(self.input_shape) == 2 and self.input_shape[1] < 1000
        )
        
        if self.uses_spectrogram:
            logger.info("Model expects spectrogram input (96x64)")
        else:
            logger.info("Model expects raw audio input")
    
    def resample_audio(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio from original sample rate to target sample rate."""
        if orig_sr == target_sr:
            return audio
        
        # Calculate new length
        new_length = int(len(audio) * target_sr / orig_sr)
        
        # Use scipy's resample for high-quality resampling
        resampled = scipy_signal.resample(audio, new_length)
        
        logger.debug(f"Resampled audio from {orig_sr}Hz ({len(audio)} samples) to {target_sr}Hz ({len(resampled)} samples)")
        return resampled.astype(np.float32)
    
    def classify(self, audio: np.ndarray, input_sample_rate: int = 16000) -> Dict:
        """
        Classify audio waveform.
        
        Args:
            audio: Audio samples as float32 numpy array, range [-1, 1]
            input_sample_rate: Sample rate of the input audio
            
        Returns:
            Dictionary with classification results
        """
        if MOCK_MODE:
            import random
            idx = random.randint(0, min(50, len(self.labels)-1))
            return {
                "top_class": self.labels[idx] if self.labels else "unknown",
                "top_score": random.randint(30, 80),
                "class_id": idx,
                "second_class": self.labels[(idx+1) % len(self.labels)],
                "second_score": random.randint(10, 30),
                "second_class_id": (idx+1) % len(self.labels),
                "all_scores": None,
            }
        
        try:
            # Resample to 16kHz if needed (YAMNet requires 16kHz)
            if input_sample_rate != 16000:
                audio = self.resample_audio(audio, input_sample_rate, 16000)
            
            # Generate spectrogram if model expects it
            if self.uses_spectrogram:
                spectrogram = self.spectrogram_gen.compute_spectrogram(audio)
                
                # Reshape to match model input shape
                # Model expects: (1, 96, 64, 1) or similar
                if len(self.input_shape) == 4:
                    input_data = spectrogram.reshape(1, 96, 64, 1)
                elif len(self.input_shape) == 3:
                    input_data = spectrogram.reshape(1, 96, 64)
                else:
                    input_data = spectrogram.flatten().reshape(self.input_shape)
                
                # Quantize if model expects int8
                if self.input_dtype == np.int8:
                    # Scale from float to int8 range
                    input_data = np.clip(input_data * 127, -128, 127).astype(np.int8)
                elif self.input_dtype == np.uint8:
                    # Scale from float to uint8 range
                    input_data = np.clip((input_data + 10) * 12.75, 0, 255).astype(np.uint8)
                else:
                    input_data = input_data.astype(self.input_dtype)
                
                logger.debug(f"Spectrogram - shape: {spectrogram.shape}, min: {spectrogram.min():.2f}, max: {spectrogram.max():.2f}")
            else:
                # Raw audio input
                required_samples = 15600  # YAMNet default
                if len(audio) < required_samples:
                    audio = np.pad(audio, (0, required_samples - len(audio)))
                else:
                    audio = audio[:required_samples]
                
                input_data = audio.astype(np.float32)
                if len(self.input_shape) == 2:
                    input_data = input_data.reshape(1, -1)
                
                logger.debug(f"Input audio - len: {len(audio)}, min: {audio.min():.4f}, max: {audio.max():.4f}")
            
            # Run inference
            self.interpreter.set_tensor(self.input_index, input_data)
            self.interpreter.invoke()
            
            # Get output scores
            output_data = self.interpreter.get_tensor(self.output_index)
            
            # Handle different output formats
            if len(output_data.shape) == 3:
                # Shape: (1, num_frames, num_classes) - average across frames
                scores = np.mean(output_data[0], axis=0)
            elif len(output_data.shape) == 2:
                scores = output_data[0]
            else:
                scores = output_data.flatten()
            
            # Convert from quantized scores if needed
            if scores.dtype in [np.int8, np.uint8]:
                scores = scores.astype(np.float32) / 255.0
            
            # Get top results
            top_indices = np.argsort(scores)[::-1][:5]
            
            result = {
                "top_class": self.labels[top_indices[0]] if top_indices[0] < len(self.labels) else f"class_{top_indices[0]}",
                "top_score": int(scores[top_indices[0]] * 100),
                "class_id": int(top_indices[0]),
            }
            
            if len(top_indices) > 1:
                result["second_class"] = self.labels[top_indices[1]] if top_indices[1] < len(self.labels) else f"class_{top_indices[1]}"
                result["second_score"] = int(scores[top_indices[1]] * 100)
                result["second_class_id"] = int(top_indices[1])
            
            # Include top 5 for debugging
            top5 = []
            for idx in top_indices[:5]:
                label = self.labels[idx] if idx < len(self.labels) else f"class_{idx}"
                top5.append(f"{label}:{scores[idx]*100:.1f}%")
            result["top5"] = top5
            
            logger.debug(f"Top 5: {', '.join(top5)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Classification error: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}


class PredictionSmoother:
    """Tracks predictions over time to reduce false positives."""
    
    def __init__(self, window_size: int = 2, min_confidence: int = 30, min_gap: int = 10,
                 classes_of_interest: List[str] = None):
        """
        Initialize prediction smoother.
        
        Args:
            window_size: Number of consecutive predictions needed
            min_confidence: Minimum confidence score to consider
            min_gap: Minimum gap between top and second prediction
            classes_of_interest: Only report these classes (empty = all)
        """
        self.window_size = window_size
        self.min_confidence = min_confidence
        self.min_gap = min_gap
        self.classes_of_interest = classes_of_interest or []
        self.history = []
        self.last_reported_class = None
        self.last_reported_time = 0
        self.cooldown_seconds = 3.0
    
    def add_prediction(self, classification: Dict) -> Optional[Dict]:
        """
        Add a prediction and return smoothed result if confident enough.
        
        Args:
            classification: Classification result from classifier
            
        Returns:
            Smoothed classification if confident, None otherwise
        """
        if "error" in classification:
            return None
        
        top_class = classification["top_class"]
        top_score = classification["top_score"]
        second_score = classification.get("second_score", 0)
        confidence_gap = top_score - second_score
        
        # Check if class is of interest
        if self.classes_of_interest:
            # Case-insensitive match
            is_interesting = any(
                c.lower() in top_class.lower() or top_class.lower() in c.lower()
                for c in self.classes_of_interest
            )
            if not is_interesting:
                return None
        
        # Check confidence
        is_confident = (
            top_score >= self.min_confidence and
            confidence_gap >= self.min_gap
        )
        
        self.history.append({
            "class": top_class,
            "score": top_score,
            "confident": is_confident,
            "full_result": classification,
            "time": time.time()
        })
        
        # Keep only recent history
        if len(self.history) > self.window_size * 2:
            self.history = self.history[-self.window_size:]
        
        # Check for consistent confident predictions
        if len(self.history) >= self.window_size:
            recent = self.history[-self.window_size:]
            
            if not all(p["confident"] for p in recent):
                return None
            
            # Check if all predictions are similar (YAMNet might give related classes)
            classes = [p["class"] for p in recent]
            # For YAMNet, check if the main category is consistent
            # e.g., "Speech", "Child speech", "Narration" are all speech
            primary_class = classes[-1]
            
            # Simple check: same class or all contain similar keywords
            if len(set(classes)) > 1:
                # Allow if they share common words
                words_in_primary = set(primary_class.lower().split())
                all_similar = True
                for c in classes:
                    words_in_c = set(c.lower().split())
                    if not words_in_primary.intersection(words_in_c):
                        all_similar = False
                        break
                if not all_similar:
                    return None
            
            # Check cooldown
            current_time = time.time()
            if (primary_class == self.last_reported_class and
                current_time - self.last_reported_time < self.cooldown_seconds):
                return None
            
            self.last_reported_class = primary_class
            self.last_reported_time = current_time
            
            avg_score = sum(p["score"] for p in recent) // len(recent)
            
            result = recent[-1]["full_result"].copy()
            result["top_score"] = avg_score
            result["smoothed"] = True
            result["consecutive_count"] = self.window_size
            
            self.history = []
            
            return result
        
        return None


class UDPAudioReceiver:
    """Receives audio data from UDP stream."""
    
    def __init__(self, name: str, port: int, sample_rate: int = 16000,
                 samples_per_packet: int = 256):
        self.name = name
        self.port = port
        self.sample_rate = sample_rate
        self.samples_per_packet = samples_per_packet
        self.socket = None
        self.audio_buffer = np.array([], dtype=np.float32)
        self.packets_received = 0
        self.continuous_raw_file = None
        self.debug_chunk_count = 0
        
        if DEBUG_AUDIO:
            os.makedirs(DEBUG_AUDIO_PATH, exist_ok=True)
            self.continuous_raw_path = f"{DEBUG_AUDIO_PATH}/{self.name}_continuous.raw"
            self.continuous_raw_file = open(self.continuous_raw_path, 'wb')
            logger.info(f"Recording to: {self.continuous_raw_path}")
    
    @staticmethod
    def calculate_rms(audio: np.ndarray) -> float:
        if len(audio) == 0:
            return 0.0
        return float(np.sqrt(np.mean(audio ** 2)))
    
    def _save_debug_audio(self, audio: np.ndarray, triggered: bool = False):
        """Save audio for debugging."""
        if not DEBUG_AUDIO:
            return
        
        audio_int16 = (audio * 32767).astype(np.int16)
        
        if triggered:
            wav_path = f"{DEBUG_AUDIO_PATH}/{self.name}_triggered_{self.debug_chunk_count:04d}.wav"
            with wave.open(wav_path, 'w') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_int16.tobytes())
            
            logger.info(f"Saved: {wav_path} ({len(audio)} samples)")
            self.debug_chunk_count += 1
    
    def start(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(("", self.port))
        logger.info(f"Listening on port {self.port}")
    
    def receive_packet(self, timeout: float = 0.05) -> Optional[np.ndarray]:
        self.socket.settimeout(timeout)
        try:
            data, addr = self.socket.recvfrom(1024)
            self.packets_received += 1
            
            num_samples = len(data) // 2
            samples = struct.unpack(f"<{num_samples}h", data)
            audio = np.array(samples, dtype=np.float32) / 32768.0
            
            # Write to continuous debug file
            if DEBUG_AUDIO and self.continuous_raw_file:
                audio_int16 = np.array(samples, dtype=np.int16)
                self.continuous_raw_file.write(audio_int16.tobytes())
                self.continuous_raw_file.flush()
            
            return audio
        except socket.timeout:
            return None
        except Exception as e:
            logger.error(f"Error receiving: {e}")
            return None
    
    def get_buffered_audio(self, num_samples: int, receive_timeout: float = 0.05) -> Optional[np.ndarray]:
        max_iterations = num_samples // self.samples_per_packet + 10
        iterations = 0
        
        while len(self.audio_buffer) < num_samples and iterations < max_iterations:
            packet = self.receive_packet(timeout=receive_timeout)
            if packet is not None:
                self.audio_buffer = np.concatenate([self.audio_buffer, packet])
            iterations += 1
        
        if len(self.audio_buffer) >= num_samples:
            audio = self.audio_buffer[:num_samples]
            self.audio_buffer = self.audio_buffer[num_samples:]
            return audio
        
        return None
    
    def stop(self):
        if self.socket:
            self.socket.close()
        if self.continuous_raw_file:
            self.continuous_raw_file.close()
            logger.info(f"Saved: {self.continuous_raw_path}")
            logger.info(f"Play: sox -t raw -b 16 -e signed -c 1 -r {self.sample_rate} {self.continuous_raw_path} -d")


def main():
    """Main application loop."""
    model_path = CONFIG["MODEL_PATH"]
    labels_path = CONFIG["LABELS_PATH"]
    
    if not Path(model_path).exists():
        logger.error(f"Model not found: {model_path}")
        logger.info("Download with: curl -L -o yamnet_edgetpu.tflite https://github.com/google-coral/test_data/raw/master/yamnet_edgetpu.tflite")
        return
    
    if not Path(labels_path).exists():
        logger.error(f"Labels not found: {labels_path}")
        logger.info("Download with: curl -L -o yamnet_class_map.csv https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv")
        return
    
    listener_configs = CONFIG.get("LISTENERS", [{"name": "default", "port": 50005}])
    
    try:
        classifier = YAMNetClassifier(model_path, labels_path)
        
        # Calculate input buffer size based on input sample rate
        # YAMNet spectrogram requires 96 frames * 160 hop + 400 window = 15600 samples at 16kHz
        input_sr = CONFIG["INPUT_SAMPLE_RATE"]
        yamnet_sr = CONFIG["YAMNET_SAMPLE_RATE"]
        # Fixed duration for spectrogram model
        required_samples_16khz = 15600  # 96*160 + 400 - 160
        input_duration = required_samples_16khz / yamnet_sr  # ~0.975s
        input_buffer_samples = int(input_duration * input_sr)
        
        CONFIG["AUDIO_BUFFER_SECONDS"] = input_duration
        
        receivers = []
        for listener in listener_configs:
            receiver = UDPAudioReceiver(
                name=listener["name"],
                port=listener["port"],
                sample_rate=input_sr,
                samples_per_packet=CONFIG["SAMPLES_PER_PACKET"]
            )
            receiver.start()
            receivers.append(receiver)
        
        logger.info("=" * 60)
        logger.info("YAMNet Sound Classifier Started")
        logger.info("=" * 60)
        logger.info(f"Input sample rate: {input_sr} Hz (from ESP)")
        logger.info(f"YAMNet sample rate: {yamnet_sr} Hz (model expects)")
        if input_sr != yamnet_sr:
            logger.info(f"  -> Will resample {input_sr} -> {yamnet_sr} Hz")
        logger.info(f"Audio buffer: {CONFIG['AUDIO_BUFFER_SECONDS']:.3f}s ({input_buffer_samples} input samples)")
        logger.info(f"Confidence threshold: {CONFIG['CERTAINTY_THRESHOLD']}%")
        logger.info(f"Ambient threshold: {CONFIG['AMBIENT_THRESHOLD']:.4f} RMS")
        logger.info(f"Smoothing window: {CONFIG['SMOOTHING_WINDOW']} predictions")
        
        if CONFIG["CLASSES_OF_INTEREST"]:
            logger.info(f"Classes of interest: {CONFIG['CLASSES_OF_INTEREST']}")
        else:
            logger.info("Reporting all detected classes")
        
        if DEBUG_AUDIO:
            logger.info(f"Debug audio: {DEBUG_AUDIO_PATH}")
        logger.info("=" * 60)
        
        smoothers = {
            listener["name"]: PredictionSmoother(
                window_size=CONFIG["SMOOTHING_WINDOW"],
                min_confidence=CONFIG["CERTAINTY_THRESHOLD"],
                min_gap=CONFIG["MIN_CONFIDENCE_GAP"],
                classes_of_interest=CONFIG["CLASSES_OF_INTEREST"],
            )
            for listener in listener_configs
        }
        
        last_level_log = 0
        
        while True:
            for receiver in receivers:
                audio = receiver.get_buffered_audio(
                    num_samples=input_buffer_samples,
                    receive_timeout=0.05,
                )
                
                if audio is None:
                    continue
                
                rms_level = receiver.calculate_rms(audio)
                
                # Periodic level logging
                current_time = time.time()
                if current_time - last_level_log > 5.0:
                    logger.info(f"[{receiver.name}] RMS: {rms_level:.4f}")
                    last_level_log = current_time
                
                # Skip quiet audio
                if rms_level < CONFIG["AMBIENT_THRESHOLD"]:
                    continue
                
                logger.debug(f"[{receiver.name}] Triggered! RMS: {rms_level:.4f}")
                
                # Save debug audio
                if DEBUG_AUDIO:
                    receiver._save_debug_audio(audio, triggered=True)
                
                # Classify with YAMNet (handles resampling internally)
                results = classifier.classify(audio, input_sample_rate=input_sr)
                
                if "error" not in results:
                    top_class = results["top_class"]
                    top_score = results["top_score"]
                    second_class = results.get("second_class", "N/A")
                    second_score = results.get("second_score", 0)
                    
                    # Log raw prediction
                    logger.debug(
                        f"[{receiver.name}] {top_class} ({top_score}%) | "
                        f"{second_class} ({second_score}%)"
                    )
                    
                    # Smooth predictions
                    smoother = smoothers[receiver.name]
                    smoothed = smoother.add_prediction(results)
                    
                    if smoothed:
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        logger.warning(
                            f"🔊 [{timestamp}] [{receiver.name}] "
                            f"{smoothed['top_class'].upper()} ({smoothed['top_score']}%)"
                        )
                else:
                    logger.error(f"[{receiver.name}] Error: {results['error']}")
    
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        for receiver in locals().get("receivers", []):
            receiver.stop()
        logger.info("Stopped")


if __name__ == "__main__":
    main()
