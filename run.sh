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

# Create debug audio directory if needed
DEBUG_AUDIO_DIR="$(pwd)/debug_audio"
mkdir -p "$DEBUG_AUDIO_DIR"

# Parse arguments
DEBUG_ENV=""
THRESHOLD_ENV=""
CONFIDENCE_ENV=""
GAP_ENV=""
SMOOTHING_ENV=""
BUFFER_ENV=""
MODEL_ENV=""
CLASSES_ENV=""
SAMPLE_RATE_ENV=""
USE_ORIGINAL=""

# Show help
if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    echo "Usage: ./run.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --build              Rebuild the Docker image"
    echo "  --debug              Enable debug mode (save audio to ./debug_audio/)"
    echo "  --original           Use original model instead of YAMNet"
    echo "  --rate=N             Input sample rate from ESP (default: 16000, or 22050)"
    echo "  --threshold=N        Ambient RMS threshold (default: 0.01)"
    echo "  --confidence=N       Minimum confidence % (default: 30)"
    echo "  --gap=N              Min gap between top predictions (default: 10)"
    echo "  --smoothing=N        Consecutive predictions needed (default: 2)"
    echo "  --buffer=N           Audio buffer in seconds (default: auto)"
    echo "  --classes=a,b,c      Only report these classes (e.g. Speech,Music,Dog)"
    echo ""
    echo "YAMNet Classes (521 total, common ones):"
    echo "  Speech, Music, Dog, Cat, Alarm, Siren, Gunshot, Glass,"
    echo "  Laughter, Crying, Cough, Sneeze, Clapping, Doorbell, etc."
    echo ""
    echo "Example: ./run.sh --debug --rate=22050 --classes=Speech,Music,Dog"
    exit 0
fi

for arg in "$@"; do
    case $arg in
        --debug)
            echo -e "${YELLOW}Debug mode enabled - audio will be saved to ./debug_audio/${NC}"
            DEBUG_ENV="-e DEBUG_AUDIO=true"
            ;;
        --original)
            echo -e "${YELLOW}Using original model (not YAMNet)${NC}"
            USE_ORIGINAL="yes"
            ;;
        --rate=*)
            RATE="${arg#*=}"
            echo -e "${YELLOW}Input sample rate: ${RATE} Hz${NC}"
            SAMPLE_RATE_ENV="-e INPUT_SAMPLE_RATE=$RATE"
            ;;
        --threshold=*)
            THRESHOLD="${arg#*=}"
            echo -e "${YELLOW}Ambient threshold: ${THRESHOLD}${NC}"
            THRESHOLD_ENV="-e AMBIENT_THRESHOLD=$THRESHOLD"
            ;;
        --confidence=*)
            CONFIDENCE="${arg#*=}"
            echo -e "${YELLOW}Min confidence: ${CONFIDENCE}%${NC}"
            CONFIDENCE_ENV="-e CERTAINTY_THRESHOLD=$CONFIDENCE"
            ;;
        --gap=*)
            GAP="${arg#*=}"
            echo -e "${YELLOW}Min confidence gap: ${GAP}%${NC}"
            GAP_ENV="-e MIN_CONFIDENCE_GAP=$GAP"
            ;;
        --smoothing=*)
            SMOOTHING="${arg#*=}"
            echo -e "${YELLOW}Smoothing window: ${SMOOTHING} predictions${NC}"
            SMOOTHING_ENV="-e SMOOTHING_WINDOW=$SMOOTHING"
            ;;
        --buffer=*)
            BUFFER="${arg#*=}"
            echo -e "${YELLOW}Audio buffer: ${BUFFER}s${NC}"
            BUFFER_ENV="-e AUDIO_BUFFER_SECONDS=$BUFFER"
            ;;
        --classes=*)
            CLASSES="${arg#*=}"
            echo -e "${YELLOW}Classes of interest: ${CLASSES}${NC}"
            CLASSES_ENV="-e CLASSES_OF_INTEREST=$CLASSES"
            ;;
    esac
done

echo -e "${GREEN}Starting container...${NC}"

# Set the command based on model choice
if [[ "$USE_ORIGINAL" == "yes" ]]; then
    CMD="python3 main.py"
else
    CMD="python3 main_yamnet.py"
fi

# Run with USB device access for Coral
docker run -it --rm \
    --name $CONTAINER_NAME \
    --privileged \
    -v /dev/bus/usb:/dev/bus/usb \
    --network host \
    -v "$(pwd)/sound_edgetpu.tflite:/app/sound_edgetpu.tflite:ro" \
    -v "$(pwd)/labels.txt:/app/labels.txt:ro" \
    -v "$(pwd)/yamnet_edgetpu.tflite:/app/yamnet_edgetpu.tflite:ro" \
    -v "$(pwd)/yamnet_class_map.csv:/app/yamnet_class_map.csv:ro" \
    -v "$DEBUG_AUDIO_DIR:/app/debug_audio" \
    $DEBUG_ENV \
    $THRESHOLD_ENV \
    $CONFIDENCE_ENV \
    $GAP_ENV \
    $SMOOTHING_ENV \
    $BUFFER_ENV \
    $CLASSES_ENV \
    $SAMPLE_RATE_ENV \
    $IMAGE_NAME $CMD
