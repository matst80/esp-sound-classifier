#!/bin/bash

# Build and run the ESP Sound Classifier with Coral USB support

IMAGE_NAME="esp-sound-classifier"
CONTAINER_NAME="sound-classifier"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}ESP Sound Classifier - Coral USB Edition${NC}"
echo "==========================================="

# Check if Coral USB is connected
if ! lsusb | grep -q "Google Inc\|Global Unichip Corp"; then
    echo -e "${YELLOW}Warning: Coral USB device not detected. Make sure it's plugged in.${NC}"
fi

# Build the image if it doesn't exist or if --build flag is passed
if [[ "$1" == "--build" ]] || [[ "$(docker images -q $IMAGE_NAME 2>/dev/null)" == "" ]]; then
    echo -e "${GREEN}Building Docker image...${NC}"
    docker build -t $IMAGE_NAME .
fi

# Stop and remove existing container if running
if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo "Stopping existing container..."
    docker rm -f $CONTAINER_NAME >/dev/null 2>&1
fi

echo -e "${GREEN}Starting container...${NC}"

# Run with USB device access for Coral
docker run -it --rm \
    --name $CONTAINER_NAME \
    --privileged \
    -v /dev/bus/usb:/dev/bus/usb \
    --network host \
    -v "$(pwd)/sound_edgetpu.tflite:/app/sound_edgetpu.tflite:ro" \
    -v "$(pwd)/labels.txt:/app/labels.txt:ro" \
    $IMAGE_NAME "$@"
