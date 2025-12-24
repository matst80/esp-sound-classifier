FROM debian:buster

# Fix archived Debian Buster sources
RUN sed -i 's/deb.debian.org/archive.debian.org/g' /etc/apt/sources.list && \
    sed -i 's/security.debian.org/archive.debian.org/g' /etc/apt/sources.list && \
    sed -i '/stretch-updates/d' /etc/apt/sources.list && \
    sed -i '/buster-updates/d' /etc/apt/sources.list

# Install system dependencies and tools for adding Coral repository
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    libusb-1.0-0 \
    libudev1 \
    libsndfile1 \
    usbutils \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Add Coral Edge TPU repository and GPG key
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -

# Install Edge TPU runtime and pycoral
RUN apt-get update && apt-get install -y \
    libedgetpu1-std \
    python3-pycoral \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --no-cache-dir -U pip

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Download YAMNet spectrogram model and labels if not present
# The spectrogram-input model is more readily available than the raw-audio version
RUN if [ ! -f yamnet_spectra_edgetpu.tflite ]; then \
        curl -L -o yamnet_spectra_edgetpu.tflite "https://raw.githubusercontent.com/google-coral/coralmicro/main/models/yamnet_spectra_in_edgetpu.tflite"; \
    fi && \
    if [ ! -f yamnet_class_map.csv ]; then \
        curl -L -o yamnet_class_map.csv "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"; \
    fi

# Default: run YAMNet classifier (better pretrained model)
# Use MODEL=original to run the original custom model
CMD ["python3", "main_yamnet.py"]
