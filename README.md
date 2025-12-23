# Sound Classifier with Google Coral USB

A Python application that classifies sounds in real-time using Google Coral Edge TPU and audio from UDP stream.

## Features

- **Real-time audio classification** using Google Coral USB accelerator
- **UDP audio streaming** support (16-bit PCM, mono, 16 kHz)
- **Librosa-based feature extraction** (MFCC, chroma, mel spectrogram, spectral contrast, tonnetz)
- **Confidence scoring** with threshold filtering
- **Logging and monitoring** of classification results

## Requirements

- Python 3.7+
- Google Coral USB Accelerator
- Audio stream sending UDP packets on port 50005

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Coral Tools

For detailed instructions, see: https://coral.ai/docs/accelerator/get-started/

```bash
# macOS (using Homebrew)
brew install libusb libedgetpu

# Linux (Debian/Ubuntu)
sudo apt-get install python3-pycoral

# Or use pip
pip install pycoral
```

### 3. Prepare Model and Labels

- Place your TFLite model at `sound_edgetpu.tflite` (already in repo)
- Update `labels.txt` with your class labels (one per line)

## Configuration

Edit the `CONFIG` dictionary in `main.py`:

```python
CONFIG = {
    "UDP_BROADCAST_IP": "255.255.255.255",  # Broadcast or specific IP (sender side)
    "LISTENERS": [                           # One or many UDP listeners
        {"name": "default", "port": 50005},
        # {"name": "stream2", "port": 50006},
    ],
    "SAMPLE_RATE": 16000,                    # Hz
    "SAMPLES_PER_PACKET": 256,               # Samples per UDP packet
    "MODEL_PATH": "sound_edgetpu.tflite",    # Path to TFLite model
    "LABELS_PATH": "labels.txt",             # Path to labels file
    "CERTAINTY_THRESHOLD": 70,               # Minimum confidence %
}
```

## UDP Stream Format

The classifier expects UDP packets with:
- **IP**: `255.255.255.255` (or specific IP)
- **Port**: as configured per listener (default `50005`)
- **Format**: 16-bit PCM, mono, 16 kHz, little endian
- **Packet Size**: 256 samples (512 bytes)
- **Timing**: ~16ms per packet (256 samples / 16000 Hz)

### Example: Sending Audio via UDP

```python
import socket
import struct
import numpy as np

def send_audio_udp(audio_file, host='255.255.255.255', port=50005):
    """Send audio samples via UDP (for testing)."""
    import librosa
    
    y, sr = librosa.load(audio_file, sr=16000, mono=True)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    
    # Send in 256-sample packets
    for i in range(0, len(y), 256):
        chunk = y[i:i+256]
        if len(chunk) < 256:
            chunk = np.pad(chunk, (0, 256 - len(chunk)))
        
        # Convert to 16-bit PCM, little endian
        pcm = (chunk * 32767).astype(np.int16)
        data = struct.pack(f'<{len(pcm)}h', *pcm)
        sock.sendto(data, (host, port))
        
        # Sleep to simulate real-time (256 samples at 16kHz = ~16ms)
        import time
        time.sleep(0.016)
    
    sock.close()
```

## Usage

### Basic Usage

```bash
python main.py
```

The classifier will:
1. Initialize the Coral Edge TPU
2. Load the TFLite model and labels
3. Listen for UDP audio packets on the configured listener ports
4. Buffer incoming audio to 1-second chunks
5. Extract features from each chunk
6. Classify and log results with confidence scores

### Example Output

```
2024-01-15 10:30:45,123 - INFO - Sound classifier started. Waiting for UDP audio packets...
2024-01-15 10:30:45,124 - INFO - Listening on port 50005
2024-01-15 10:30:45,125 - INFO - Certainty threshold: 70%
2024-01-15 10:30:50,456 - INFO - [2024-01-15T10:30:50.456123] Top: speech (87%) | 2nd: music (13%)
2024-01-15 10:30:51,456 - INFO - [2024-01-15T10:30:51.456456] Top: background (65%) | 2nd: speech (35%)
2024-01-15 10:30:52,456 - WARNING - Classification above threshold: siren (92%)
```

## Architecture

### SoundClassifier
- Manages the Coral Edge TPU interpreter
- Loads and caches model and labels
- Extracts audio features using librosa
- Performs inference and returns results

### UDPAudioReceiver
- Receives UDP packets with PCM audio
- Buffers packets into audio chunks
- Handles socket timeout and errors gracefully

### Feature Extraction
Mirrors the reference implementation:
- **MFCC** (40 coefficients): Speech characteristics
- **Chroma**: Harmonic content
- **Mel Spectrogram**: Frequency distribution
- **Spectral Contrast**: Energy difference in frequency bands
- **Tonnetz**: Tonal centroid features

Total: 193 features as input to the model

## Troubleshooting

### "Is the Coral USB accelerator plugged in?"
- Ensure the USB Coral accelerator is connected
- Check with: `lsusb | grep Google` (Linux) or `system_profiler SPUSBDataType` (macOS)

### No UDP packets received
- Verify sender is using correct IP and port (50005)
- Check firewall settings: `sudo iptables -L | grep 50005` (Linux)
- Test with: `nc -u -l 50005` on receiver

### Poor classification accuracy
- Ensure audio is properly normalized (PCM 16-bit)
- Check sample rate is 16 kHz
- Verify model matches your audio domain
- Consider retraining model with your specific sound classes

### Model not found
- Ensure `sound_edgetpu.tflite` is in the working directory
- For training on custom data, see: https://coral.ai/docs/edgetpu/retrain-detection/

## References

- [Google Coral Documentation](https://coral.ai/)
- [Reference Classifier](https://github.com/balena-io-experimental/coral-audio-analysis)
- [Librosa Audio Processing](https://librosa.org/)
- [Edge TPU Python API](https://coral.ai/docs/reference/py/)

## License

MIT License
