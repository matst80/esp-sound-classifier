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
from typing import Dict, Optional

# Check if we're in mock mode (no Coral hardware)
MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"

if MOCK_MODE:
    print("Running in MOCK MODE - no Coral hardware required")
    make_interpreter = None
    classify = None
else:
    try:
        from pycoral.adapters import classify
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
    "SAMPLE_RATE": 16000,
    "SAMPLES_PER_PACKET": 256,
    "MODEL_PATH": "sound_edgetpu.tflite",
    "LABELS_PATH": "labels.txt",
    "CERTAINTY_THRESHOLD": 70,
}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
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
        
        Args:
            audio_data: Audio samples as numpy array
            sample_rate: Sample rate in Hz
            
        Returns:
            Feature vector (193 features)
        """
        if len(audio_data) == 0:
            return np.zeros((1, 193))
        
        try:
            # Normalize audio
            audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-8)
            
            # STFT for chroma and spectral contrast
            stft = np.abs(librosa.stft(audio_data))
            
            # Extract all features
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
            
            # Combine all features
            features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
            return features.reshape(1, -1)
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
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
            results = classify.classify(
                self.interpreter,
                features.astype(np.float32)
            )
            
            if not results:
                return {"error": "No classification results"}
            
            # Sort by score descending and get top 2
            sorted_results = sorted(results, key=lambda x: x.score, reverse=True)[:2]
            
            classification = {
                "top_class": self.labels[sorted_results[0].class_id],
                "top_score": int(sorted_results[0].score * 100),
                "class_id": sorted_results[0].class_id,
            }
            
            if len(sorted_results) > 1:
                classification.update({
                    "second_class": self.labels[sorted_results[1].class_id],
                    "second_score": int(sorted_results[1].score * 100),
                    "second_class_id": sorted_results[1].class_id,
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
            data, addr = self.socket.recvfrom(512)  # 256 samples * 2 bytes
            self.packets_received += 1
            
            # Unpack 16-bit PCM, little endian
            samples = struct.unpack(f"<{len(data)//2}h", data)
            audio = np.array(samples, dtype=np.float32) / 32768.0
            
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
        while len(self.audio_buffer) < num_samples:
            packet = self.receive_packet(timeout=receive_timeout)
            if packet is None:
                if len(self.audio_buffer) > 0:
                    break  # Return what we have
                else:
                    return None
            self.audio_buffer = np.concatenate([self.audio_buffer, packet])
        
        if len(self.audio_buffer) >= num_samples:
            audio = self.audio_buffer[:num_samples]
            self.audio_buffer = self.audio_buffer[num_samples:]
            return audio
        elif len(self.audio_buffer) > 0:
            audio = self.audio_buffer.copy()
            self.audio_buffer = np.array([], dtype=np.float32)
            return audio
        
        return None
    
    def stop(self):
        """Stop receiving and close socket."""
        if self.socket:
            self.socket.close()


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
        
        # Main processing loop
        while True:
            for receiver in receivers:
                audio = receiver.get_buffered_audio(
                    num_samples=CONFIG["SAMPLE_RATE"],
                    receive_timeout=0.05,
                )
                
                if audio is None:
                    continue
                
                # Extract features
                features = classifier.extract_features(audio, CONFIG["SAMPLE_RATE"])
                
                # Classify
                results = classifier.classify(features)
                
                if "error" not in results:
                    top_class = results["top_class"]
                    top_score = results["top_score"]
                    second_class = results["second_class"]
                    second_score = results["second_score"]
                    
                    timestamp = datetime.now().isoformat()
                    logger.info(
                        f"[{timestamp}] [{receiver.name}:{receiver.port}] "
                        f"Top: {top_class} ({top_score}%) | "
                        f"2nd: {second_class} ({second_score}%)"
                    )
                    
                    if top_score >= CONFIG["CERTAINTY_THRESHOLD"]:
                        logger.warning(
                            f"[{receiver.name}:{receiver.port}] "
                            f"Classification above threshold: "
                            f"{top_class} ({top_score}%)"
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
    finally:
        for receiver in locals().get("receivers", []):
            receiver.stop()
        logger.info("Sound classifier stopped")


if __name__ == "__main__":
    main()
