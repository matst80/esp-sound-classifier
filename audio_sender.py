#!/usr/bin/env python3
"""
UDP Audio Sender for testing the sound classifier.

Reads an audio file and sends it via UDP in the expected format:
- 256 samples per packet
- 16-bit PCM, mono, 16 kHz, little endian
"""

import socket
import struct
import numpy as np
import argparse
import time
from pathlib import Path

try:
    import librosa
except ImportError:
    print("Error: librosa not installed. Install with: pip install librosa")
    exit(1)


def send_audio_udp(audio_file: str, host: str = "255.255.255.255", 
                   port: int = 50005, realtime: bool = True):
    """
    Send audio samples via UDP.
    
    Args:
        audio_file: Path to audio file to send
        host: Destination host (IP address or broadcast)
        port: Destination port
        realtime: If True, send at real-time pace (256 samples ~= 16ms)
    """
    if not Path(audio_file).exists():
        print(f"Error: Audio file not found: {audio_file}")
        return
    
    # Load audio
    print(f"Loading audio from: {audio_file}")
    y, sr = librosa.load(audio_file, sr=16000, mono=True)
    print(f"Loaded {len(y)} samples at {sr} Hz")
    
    # Create UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    
    print(f"Sending to {host}:{port}")
    print(f"Realtime mode: {realtime}")
    print(f"Sending {len(y) // 256} packets of 256 samples...")
    
    packets_sent = 0
    samples_sent = 0
    
    # Send in 256-sample packets
    for i in range(0, len(y), 256):
        chunk = y[i:i + 256]
        
        # Pad if necessary
        if len(chunk) < 256:
            chunk = np.pad(chunk, (0, 256 - len(chunk)))
        
        # Normalize and convert to 16-bit PCM
        chunk_normalized = np.clip(chunk, -1.0, 1.0)
        pcm = (chunk_normalized * 32767).astype(np.int16)
        
        # Pack as little-endian 16-bit signed integers
        data = struct.pack(f'<{len(pcm)}h', *pcm)
        
        # Send packet
        sock.sendto(data, (host, port))
        packets_sent += 1
        samples_sent += len(chunk)
        
        # Print progress
        if packets_sent % 100 == 0:
            print(f"Sent {packets_sent} packets ({samples_sent} samples)")
        
        # Realtime pacing: 256 samples at 16kHz = ~16ms
        if realtime:
            time.sleep(256 / 16000.0)
    
    sock.close()
    
    print(f"\nComplete!")
    print(f"Packets sent: {packets_sent}")
    print(f"Total samples: {samples_sent}")
    print(f"Duration: {samples_sent / 16000:.2f} seconds")


def main():
    parser = argparse.ArgumentParser(
        description="Send audio via UDP for testing the sound classifier"
    )
    parser.add_argument(
        "audio_file",
        help="Path to audio file to send"
    )
    parser.add_argument(
        "--host",
        default="255.255.255.255",
        help="Destination host (default: 255.255.255.255)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=50005,
        help="Destination port (default: 50005)"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Send as fast as possible (ignores real-time pacing)"
    )
    
    args = parser.parse_args()
    
    send_audio_udp(
        args.audio_file,
        host=args.host,
        port=args.port,
        realtime=not args.fast
    )


if __name__ == "__main__":
    main()
