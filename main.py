#!/usr/bin/env python3
"""
Google Coral USB sound classifier that processes audio from UDP stream.

UDP Stream Format:
- Broadcast IP: 255.255.255.255
- Port: 50005
- Format: 16-bit PCM, mono, 16 kHz, little endian, 256 samples per packet
"""

import socket
import struct
import numpy as np
import librosa
from datetime import datetime
from pathlib import Path
import logging
import os
import wave
import time
from typing import Dict, Optional

# Check if we're in mock mode (no Coral hardware)
MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"
# Debug mode: save received audio to file for verification
DEBUG_AUDIO = os.getenv("DEBUG_AUDIO", "false").lower() == "true"
DEBUG_AUDIO_PATH = os.getenv("DEBUG_AUDIO_PATH", "/app/debug_audio")

if MOCK_MODE:
    print("Running in MOCK MODE - no Coral hardware required")
    make_interpreter = None
    common = None
else:
    try:
        from pycoral.adapters import common
        from pycoral.utils.edgetpu import make_interpreter
    except ImportError as e:
        print(f"Error importing pycoral: {e}")
        print("Trying fallback to TensorFlow Lite...")
        try:
            # Fallback: try tflite-runtime or full tensorflow
            try:
                from tflite_runtime.interpreter import Interpreter
            except ImportError:
                import tensorflow as tf
                Interpreter = tf.lite.Interpreter
            
            # Create a simple wrapper for non-Coral mode
            def make_interpreter(model_path):
                return Interpreter(model_path=model_path)
            
            classify = None  # Will use manual classification
            common = None
            print("Using TensorFlow Lite (CPU mode, no Coral acceleration)")
        except ImportError:
            print("Set MOCK_MODE=true to test without Coral hardware")
            print("Or install pycoral: pip install --extra-index-url https://google-coral.github.io/py-repo/ pycoral~=0.2")
            print("Or install tensorflow: pip install tensorflow")
            exit(1)


# Configuration
CONFIG = {
    "UDP_BROADCAST_IP": "255.255.255.255",
    # Multiple listeners are supported; each item needs a name and port.
    # Example: [{"name": "stream1", "port": 50005}, {"name": "stream2", "port": 50006}]
    "LISTENERS": [
        {"name": "default", "port": 50005},
    ],
    "SAMPLE_RATE": 22050,
    "SAMPLES_PER_PACKET": 256,
    "MODEL_PATH": "sound_edgetpu.tflite",
    "LABELS_PATH": "labels.txt",
    # Minimum confidence to consider a detection valid (0-100)
    "CERTAINTY_THRESHOLD": int(os.getenv("CERTAINTY_THRESHOLD", "50")),
    # Minimum gap between top and second prediction to consider it confident
    "MIN_CONFIDENCE_GAP": int(os.getenv("MIN_CONFIDENCE_GAP", "15")),
    # Ambient noise threshold (RMS level 0.0-1.0). Only classify when audio exceeds this.
    # Set to 0.0 to disable threshold (classify everything)
    "AMBIENT_THRESHOLD": float(os.getenv("AMBIENT_THRESHOLD", "0.002")),
    # Number of consecutive consistent predictions required before reporting
    "SMOOTHING_WINDOW": int(os.getenv("SMOOTHING_WINDOW", "1")),
    # Audio buffer size in seconds (longer = more context but more latency)
    "AUDIO_BUFFER_SECONDS": float(os.getenv("AUDIO_BUFFER_SECONDS", "1.5")),
}

# Setup logging
log_level = logging.DEBUG if DEBUG_AUDIO else logging.INFO
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SoundClassifier:
    """Handles sound classification using Google Coral Edge TPU."""
    
    def __init__(self, model_path: str, labels_path: str):
        """
        Initialize the sound classifier.
        
        Args:
            model_path: Path to the TFLite model file
            labels_path: Path to the labels file
        """
        self.model_path = model_path
        self.labels_path = labels_path
        self.labels = self._load_labels()
        self.interpreter = self._init_interpreter()
        logger.info(f"Loaded {len(self.labels)} labels")
        
    def _load_labels(self) -> list:
        """Load labels from file."""
        try:
            with open(self.labels_path, 'r') as f:
                return [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            logger.error(f"Labels file not found: {self.labels_path}")
            return []
    
    def _init_interpreter(self):
        """Initialize TFLite interpreter with Edge TPU support."""
        if MOCK_MODE:
            logger.info("MOCK MODE: Skipping Coral Edge TPU initialization")
            return None
        try:
            interpreter = make_interpreter(self.model_path)
            interpreter.allocate_tensors()
            logger.info("Initialized Coral Edge TPU interpreter")
            return interpreter
        except Exception as e:
            logger.error(f"Failed to initialize interpreter: {e}")
            logger.error("Is the Coral USB accelerator plugged in?")
            raise
    
    def extract_features(self, audio_data: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Extract audio features using librosa.
        
        This mirrors the approach from the reference classifier:
        MFCC, chroma, mel spectrogram, spectral contrast, and tonnetz features.
        
        NOTE: The reference project records at 44100 Hz and librosa.load() 
        resamples to 22050 Hz. Our ESP sends at 16000 Hz which has less 
        frequency information. We process at the native sample rate to avoid
        upsampling artifacts.
        
        Args:
            audio_data: Audio samples as numpy array
            sample_rate: Sample rate in Hz
            
        Returns:
            Feature vector (193 features)
        """
        if len(audio_data) == 0:
            return np.zeros((1, 193))
        
        try:
            # The reference uses 44100 Hz recording -> 22050 Hz via librosa.load()
            # Our ESP sends 16000 Hz. Two options:
            # 1. Process at native 16kHz (loses some accuracy but no upsampling artifacts)
            # 2. Upsample to 22050 Hz (matches reference but adds interpolated data)
            #
            # We'll try processing at native rate first since 16kHz still captures
            # most sound classification features (human hearing is up to ~8kHz for speech)
            
            # Use native sample rate - don't resample
            # This is different from reference but avoids upsampling artifacts
            
            # STFT for chroma and spectral contrast (using defaults)
            stft = np.abs(librosa.stft(audio_data))
            
            # Extract all features - matching reference exactly
            mfccs = np.mean(
                librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40).T,
                axis=0
            )
            chroma = np.mean(
                librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,
                axis=0
            )
            mel = np.mean(
                librosa.feature.melspectrogram(y=audio_data, sr=sample_rate).T,
                axis=0
            )
            contrast = np.mean(
                librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,
                axis=0
            )
            tonnetz = np.mean(
                librosa.feature.tonnetz(
                    y=librosa.effects.harmonic(audio_data),
                    sr=sample_rate
                ).T,
                axis=0
            )
            
            # Combine all features (should be 193: 40 + 12 + 128 + 7 + 6)
            features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
            
            logger.debug(f"Feature dimensions - MFCC: {mfccs.shape}, Chroma: {chroma.shape}, "
                        f"Mel: {mel.shape}, Contrast: {contrast.shape}, Tonnetz: {tonnetz.shape}, "
                        f"Total: {features.shape}")
            
            return features.reshape(1, -1)
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros((1, 193))
    
    def classify(self, features: np.ndarray) -> Dict:
        """
        Classify audio features.
        
        Args:
            features: Feature vector to classify
            
        Returns:
            Dictionary with classification results
        """
        if MOCK_MODE:
            # Return mock classification for testing
            import random
            idx1, idx2 = random.sample(range(len(self.labels)), min(2, len(self.labels)))
            score1 = random.randint(50, 95)
            score2 = 100 - score1
            return {
                "top_class": self.labels[idx1] if self.labels else "unknown",
                "top_score": score1,
                "class_id": idx1,
                "second_class": self.labels[idx2] if len(self.labels) > 1 else "N/A",
                "second_score": score2,
                "second_class_id": idx2,
            }
        
        try:
            # Set input tensor
            input_details = self.interpreter.get_input_details()
            output_details = self.interpreter.get_output_details()
            
            # Debug: log model input/output shapes on first call
            if not hasattr(self, '_logged_shapes'):
                logger.info(f"Model input shape: {input_details[0]['shape']}, dtype: {input_details[0]['dtype']}")
                logger.info(f"Model output shape: {output_details[0]['shape']}, dtype: {output_details[0]['dtype']}")
                logger.info(f"Features shape: {features.shape}")
                self._logged_shapes = True
            
            # Prepare input data
            input_data = features.astype(np.float32)
            expected_shape = tuple(input_details[0]['shape'])
            
            if input_data.shape != expected_shape:
                logger.warning(f"Feature shape mismatch: got {input_data.shape}, expected {expected_shape}")
                input_data = np.resize(input_data, expected_shape)
            
            # Debug: log feature statistics
            # logger.debug(f"Input stats - min: {input_data.min():.4f}, max: {input_data.max():.4f}, mean: {input_data.mean():.4f}")
            
            self.interpreter.set_tensor(input_details[0]['index'], input_data)
            self.interpreter.invoke()
            
            # Get output
            output_data = self.interpreter.get_tensor(output_details[0]['index'])
            scores = output_data.flatten()
            
            # Debug: log all scores
            # logger.debug(f"Raw scores: {scores}")
            
            # Get top 2 results
            top_indices = np.argsort(scores)[::-1][:2]
            
            classification = {
                "top_class": self.labels[top_indices[0]] if top_indices[0] < len(self.labels) else "unknown",
                "top_score": int(scores[top_indices[0]] * 100),
                "class_id": int(top_indices[0]),
            }
            
            if len(top_indices) > 1:
                classification.update({
                    "second_class": self.labels[top_indices[1]] if top_indices[1] < len(self.labels) else "unknown",
                    "second_score": int(scores[top_indices[1]] * 100),
                    "second_class_id": int(top_indices[1]),
                })
            else:
                classification.update({
                    "second_class": "N/A",
                    "second_score": 0,
                    "second_class_id": -1,
                })
            
            return classification
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return {"error": str(e)}


class PredictionSmoother:
    """Tracks predictions over time to reduce false positives."""
    
    def __init__(self, window_size: int = 3, min_confidence: int = 50, min_gap: int = 15):
        """
        Initialize prediction smoother.
        
        Args:
            window_size: Number of consecutive predictions needed
            min_confidence: Minimum confidence score to consider
            min_gap: Minimum gap between top and second prediction
        """
        self.window_size = window_size
        self.min_confidence = min_confidence
        self.min_gap = min_gap
        self.history = []
        self.last_reported_class = None
        self.last_reported_time = 0
        self.cooldown_seconds = 2.0  # Don't report same class within this time
    
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
        
        # Check if this prediction is confident enough
        is_confident = (
            top_score >= self.min_confidence and
            confidence_gap >= self.min_gap
        )
        
        # Add to history
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
            
            # All recent predictions must be confident
            if not all(p["confident"] for p in recent):
                return None
            
            # All recent predictions must be the same class
            classes = [p["class"] for p in recent]
            if len(set(classes)) != 1:
                return None
            
            # Check cooldown (don't spam the same detection)
            current_time = time.time()
            detected_class = classes[0]
            
            if (detected_class == self.last_reported_class and
                current_time - self.last_reported_time < self.cooldown_seconds):
                return None
            
            # We have a confident, consistent prediction!
            self.last_reported_class = detected_class
            self.last_reported_time = current_time
            
            # Average the scores
            avg_score = sum(p["score"] for p in recent) // len(recent)
            
            result = recent[-1]["full_result"].copy()
            result["top_score"] = avg_score
            result["smoothed"] = True
            result["consecutive_count"] = self.window_size
            
            # Clear history after reporting
            self.history = []
            
            return result
        
        return None
    
    def reset(self):
        """Reset the prediction history."""
        self.history = []


class UDPAudioReceiver:
    """Receives audio data from UDP stream."""
    
    def __init__(self, name: str, port: int, sample_rate: int = 16000,
                 samples_per_packet: int = 256):
        """
        Initialize UDP receiver.
        
        Args:
            name: Label for this stream (used in logs)
            port: UDP port to listen on
            sample_rate: Expected sample rate
            samples_per_packet: Samples per UDP packet
        """
        self.name = name
        self.port = port
        self.sample_rate = sample_rate
        self.samples_per_packet = samples_per_packet
        self.socket = None
        self.audio_buffer = np.array([], dtype=np.float32)
        self.packets_received = 0
        self.debug_file = None
        self.debug_chunk_count = 0
        self.continuous_debug_file = None
        self.continuous_raw_file = None
        
        # Setup debug audio output
        if DEBUG_AUDIO:
            os.makedirs(DEBUG_AUDIO_PATH, exist_ok=True)
            logger.info(f"Debug audio enabled. Files will be saved to {DEBUG_AUDIO_PATH}")
            # Create a continuous raw file for the entire session
            # This can be played with: sox -t raw -b 16 -e signed -c 1 -r 16000 file.raw -d
            self.continuous_raw_path = f"{DEBUG_AUDIO_PATH}/{self.name}_continuous.raw"
            self.continuous_raw_file = open(self.continuous_raw_path, 'wb')
            logger.info(f"Recording continuous audio to: {self.continuous_raw_path}")
    
    def _append_to_continuous(self, audio: np.ndarray):
        """Append audio to the continuous debug file."""
        if self.continuous_raw_file is None:
            return
        try:
            # Convert float32 back to int16 for saving
            audio_int16 = (audio * 32767).astype(np.int16)
            self.continuous_raw_file.write(audio_int16.tobytes())
            self.continuous_raw_file.flush()
        except Exception as e:
            logger.debug(f"Error appending to continuous raw: {e}")
    
    @staticmethod
    def calculate_rms(audio: np.ndarray) -> float:
        """Calculate RMS (root mean square) level of audio."""
        if len(audio) == 0:
            return 0.0
        return float(np.sqrt(np.mean(audio ** 2)))
    
    @staticmethod
    def calculate_peak(audio: np.ndarray) -> float:
        """Calculate peak level of audio."""
        if len(audio) == 0:
            return 0.0
        return float(np.max(np.abs(audio)))
    
    def _save_debug_audio(self, audio: np.ndarray, triggered: bool = False):
        """Save audio chunk to debug file for verification."""
        if not DEBUG_AUDIO:
            return
        
        # Convert float32 back to int16 for saving
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Append to continuous raw file
        self._append_to_continuous(audio)
        
        # Only save individual chunks when triggered
        if triggered:
            # Save as raw file (compatible with: sox -t raw -b 16 -e signed -c 1 -r 16000 file.raw -d)
            raw_path = f"{DEBUG_AUDIO_PATH}/{self.name}_triggered_{self.debug_chunk_count:04d}.raw"
            audio_int16.tofile(raw_path)
            
            # Also save as WAV for easier playback
            wav_path = f"{DEBUG_AUDIO_PATH}/{self.name}_triggered_{self.debug_chunk_count:04d}.wav"
            with wave.open(wav_path, 'w') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_int16.tobytes())
            
            rms = self.calculate_rms(audio)
            logger.info(f"Saved triggered audio: {wav_path} ({len(audio)} samples, RMS: {rms:.4f})")
            self.debug_chunk_count += 1
            
            # Keep only last 20 triggered chunks
            if self.debug_chunk_count > 20:
                old_raw = f"{DEBUG_AUDIO_PATH}/{self.name}_triggered_{self.debug_chunk_count - 21:04d}.raw"
                old_wav = f"{DEBUG_AUDIO_PATH}/{self.name}_triggered_{self.debug_chunk_count - 21:04d}.wav"
                try:
                    if os.path.exists(old_raw):
                        os.remove(old_raw)
                    if os.path.exists(old_wav):
                        os.remove(old_wav)
                except Exception:
                    pass
    
    def start(self):
        """Start listening for UDP packets."""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(("", self.port))
        logger.info(f"Listening for UDP audio on port {self.port}")
    
    def receive_packet(self, timeout: float = 0.05) -> Optional[np.ndarray]:
        """
        Receive a single UDP audio packet.
        
        Args:
            timeout: Socket timeout in seconds
            
        Returns:
            Audio samples as numpy array or None on timeout
        """
        self.socket.settimeout(timeout)
        try:
            data, addr = self.socket.recvfrom(1024)  # Allow larger packets
            self.packets_received += 1
            
            # Unpack 16-bit PCM, little endian
            num_samples = len(data) // 2
            samples = struct.unpack(f"<{num_samples}h", data)
            audio = np.array(samples, dtype=np.float32) / 32768.0
            
            # Write raw packet immediately to continuous debug file
            if DEBUG_AUDIO and self.continuous_raw_file:
                audio_int16 = np.array(samples, dtype=np.int16)
                self.continuous_raw_file.write(audio_int16.tobytes())
                self.continuous_raw_file.flush()
            
            return audio
        except socket.timeout:
            return None
        except Exception as e:
            logger.error(f"Error receiving packet: {e}")
            return None
    
    def get_buffered_audio(self, num_samples: int = 16000,
                           receive_timeout: float = 0.05) -> Optional[np.ndarray]:
        """
        Get buffered audio of specified length.
        
        Args:
            num_samples: Number of samples to retrieve
            
        Returns:
            Audio buffer or None if not enough data
        """
        # Keep receiving until we have enough samples
        # Use a max iteration count to prevent infinite loop
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
        elif len(self.audio_buffer) > 0:
            # Not enough data yet, return None and keep buffer
            return None
        
        return None
    
    def stop(self):
        """Stop receiving and close socket."""
        if self.socket:
            self.socket.close()
        if self.continuous_raw_file:
            self.continuous_raw_file.close()
            logger.info(f"Continuous audio saved to: {self.continuous_raw_path}")
            logger.info(f"Play with: sox -t raw -b 16 -e signed -c 1 -r 22050 {self.continuous_raw_path} -d")


def main():
    """Main application loop."""
    # Verify model and labels exist
    if not Path(CONFIG["MODEL_PATH"]).exists():
        logger.error(f"Model file not found: {CONFIG['MODEL_PATH']}")
        return
    
    if not Path(CONFIG["LABELS_PATH"]).exists():
        logger.error(f"Labels file not found: {CONFIG['LABELS_PATH']}")
        return
    
    # Fallback: allow legacy single-port config
    listener_configs = CONFIG.get("LISTENERS") or [
        {"name": "default", "port": CONFIG.get("UDP_PORT", 50005)}
    ]
    
    try:
        classifier = SoundClassifier(CONFIG["MODEL_PATH"], CONFIG["LABELS_PATH"])
        receivers = []
        for listener in listener_configs:
            receiver = UDPAudioReceiver(
                name=listener["name"],
                port=listener["port"],
                sample_rate=CONFIG["SAMPLE_RATE"],
                samples_per_packet=CONFIG["SAMPLES_PER_PACKET"]
            )
            receiver.start()
            receivers.append(receiver)
            logger.info(f"Listening on '{listener['name']}' port {listener['port']}")
        
        logger.info("Sound classifier started. Waiting for UDP audio packets...")
        logger.info(f"Certainty threshold: {CONFIG['CERTAINTY_THRESHOLD']}%")
        logger.info(f"Ambient threshold: {CONFIG['AMBIENT_THRESHOLD']:.4f} RMS")
        if DEBUG_AUDIO:
            logger.info(f"DEBUG MODE: Audio will be saved to {DEBUG_AUDIO_PATH}")
            logger.info(f"  Continuous stream: sox -t raw -b 16 -e signed -c 1 -r 22050 {DEBUG_AUDIO_PATH}/<name>_continuous.raw -d")
            logger.info(f"  Triggered chunks: {DEBUG_AUDIO_PATH}/<name>_triggered_XXXX.wav")
        
        # Calculate audio buffer size
        audio_buffer_samples = int(CONFIG["SAMPLE_RATE"] * CONFIG["AUDIO_BUFFER_SECONDS"])
        logger.info(f"Audio buffer: {CONFIG['AUDIO_BUFFER_SECONDS']}s ({audio_buffer_samples} samples)")
        logger.info(f"Smoothing window: {CONFIG['SMOOTHING_WINDOW']} predictions")
        logger.info(f"Min confidence gap: {CONFIG['MIN_CONFIDENCE_GAP']}%")
        
        # Create prediction smoothers for each receiver
        smoothers = {
            listener["name"]: PredictionSmoother(
                window_size=CONFIG["SMOOTHING_WINDOW"],
                min_confidence=CONFIG["CERTAINTY_THRESHOLD"],
                min_gap=CONFIG["MIN_CONFIDENCE_GAP"]
            )
            for listener in listener_configs
        }
        
        # Main processing loop
        last_level_log = 0
        while True:
            for receiver in receivers:
                audio = receiver.get_buffered_audio(
                    num_samples=audio_buffer_samples,
                    receive_timeout=0.05,
                )
                
                if audio is None:
                    continue
                
                # Calculate audio level
                rms_level = receiver.calculate_rms(audio)
                peak_level = receiver.calculate_peak(audio)
                
                # Log audio levels periodically (every ~5 seconds)
                current_time = time.time()
                if current_time - last_level_log > 5.0:
                    logger.info(f"[{receiver.name}] Audio level - RMS: {rms_level:.4f}, Peak: {peak_level:.4f}")
                    last_level_log = current_time
                
                # Check if audio exceeds ambient threshold
                triggered = rms_level > CONFIG["AMBIENT_THRESHOLD"]
                
                # Save triggered chunks for debugging
                if DEBUG_AUDIO and triggered:
                    receiver._save_debug_audio(audio, triggered=True)
                
                # Only classify if above ambient threshold
                if not triggered:
                    continue
                
                logger.debug(f"[{receiver.name}] Triggered! RMS: {rms_level:.4f} > threshold {CONFIG['AMBIENT_THRESHOLD']:.4f}")
                
                # Extract features
                features = classifier.extract_features(audio, CONFIG["SAMPLE_RATE"])
                
                # Classify
                results = classifier.classify(features)
                
                if "error" not in results:
                    top_class = results["top_class"]
                    top_score = results["top_score"]
                    second_class = results["second_class"]
                    second_score = results["second_score"]
                    confidence_gap = top_score - second_score
                    
                    # Log raw prediction at debug level
                    logger.debug(
                        f"[{receiver.name}] Raw: {top_class} ({top_score}%) | "
                        f"2nd: {second_class} ({second_score}%) | Gap: {confidence_gap}%"
                    )
                    
                    # Add to smoother
                    smoother = smoothers[receiver.name]
                    smoothed_result = smoother.add_prediction(results)
                    
                    if smoothed_result:
                        # We have a confident, consistent detection!
                        detected_class = smoothed_result["top_class"]
                        detected_score = smoothed_result["top_score"]
                        timestamp = datetime.now().isoformat()
                        
                        logger.warning(
                            f"🔊 DETECTED [{timestamp}] [{receiver.name}] "
                            f"{detected_class.upper()} ({detected_score}%) "
                            f"[{smoothed_result['consecutive_count']} consistent predictions]"
                        )
                else:
                    logger.error(
                        f"[{receiver.name}:{receiver.port}] "
                        f"Classification failed: {results['error']}"
                    )
    
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        for receiver in locals().get("receivers", []):
            receiver.stop()
        logger.info("Sound classifier stopped")


if __name__ == "__main__":
    main()
