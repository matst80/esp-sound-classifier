FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libusb-1.0-0 \
    libudev1 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Edge TPU runtime
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list \
    && apt-get update --allow-insecure-repositories \
    && apt-get install -y --allow-unauthenticated libedgetpu1-std \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install pycoral
RUN pip install --no-cache-dir --extra-index-url https://google-coral.github.io/py-repo/ pycoral~=0.2

# Copy application files
COPY . .

# Run the classifier
CMD ["python", "main.py"]
