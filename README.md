# 🤖 Autonomous Line-Following Robot with Obstacle Avoidance

A Raspberry Pi 4-based autonomous robot that follows black lines, detects and avoids colored obstacles, and performs U-turns based on QR code commands.

## 📋 Table of Contents

- [Features](#features)
- [Hardware Requirements](#hardware-requirements)
- [Software Requirements](#software-requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## ✨ Features

- **Line Following**: Tracks black lines using computer vision with HSV color masking
- **Obstacle Detection**: Uses ultrasonic sensor to detect obstacles at 23cm threshold
- **Color-Based Avoidance**: 
  - 🔴 Red obstacles → Turn right
  - 🟢 Green obstacles → Turn left
- **YOLO Object Detection**: Custom-trained ONNX model for accurate color classification
- **QR Code Commands**: Scan QR codes to trigger U-turn maneuvers
  - `Ot2R1yXg` → Reverse and turn left
  - `V1uzCDL3` → Reverse and turn right
- **Live Video Stream**: Web interface to monitor robot's vision at `http://<robot-ip>:5000`
- **Smart Search Pattern**: Oscillating search when line is lost

## 🔧 Hardware Requirements

### Core Components
- **Raspberry Pi 4** (4GB+ recommended)
- **AUPPBot Motor Controller** (connected via `/dev/ttyUSB0`)
- **4 DC Motors** (tank drive configuration)
- **Raspberry Pi Camera Module** or USB webcam

### Sensors
- **HC-SR04 Ultrasonic Sensor**
  - TRIG → GPIO 21
  - ECHO → GPIO 20
  - VCC → 5V
  - GND → GND

### Power
- Battery pack capable of powering Raspberry Pi and motors
- Separate power rail recommended for motors

## 💻 Software Requirements

### Operating System
- Raspberry Pi OS (Bullseye or later)

### Python Libraries
```bash
# Core dependencies
python3-opencv
numpy
flask
RPi.GPIO
pyzbar
ultralytics  # For YOLO inference
```

### Custom Library
- `auppbot` - Motor controller library for AUPPBot hardware

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/line-following-robot.git
cd line-following-robot
```

### 2. Install System Dependencies
```bash
sudo apt update
sudo apt install -y python3-opencv python3-pip libzbar0
```

### 3. Install Python Packages
```bash
pip3 install numpy flask RPi.GPIO pyzbar ultralytics
```

### 4. Install AUPPBot Library
```bash
# Follow your AUPPBot installation instructions
pip3 install auppbot
```

### 5. Place YOLO Model
Ensure your trained ONNX model is located at:
```bash
/home/aupp/Documents/Documents/afternoonProj1Group5/red-green-box.onnx
```
Or update `MODEL_PATH` in the configuration section of the script.

## ⚙️ Configuration

Edit the configuration section at the top of `main.py`:

```python
# Camera settings
CAM_INDEX = 0          # Camera device index
W, H = 640, 480        # Resolution
ROTATE_90_CW = True    # Rotate camera 180°

# Motor control
BASE = 15              # Base forward speed (PWM)
DELTA = 8              # Steering adjustment
SEARCH_SPIN = 18       # Search pattern spin speed

# Ultrasonic sensor
STOP_THRESHOLD_CM = 23.0  # Obstacle detection distance

# YOLO model
MODEL_PATH = "/path/to/red-green-box.onnx"
YOLO_CONF = 0.5        # Detection confidence threshold
```

### Motor Direction Calibration
If motors run in wrong direction, adjust:
```python
LEFT_SIGN = +1   # Change to -1 to reverse left side
RIGHT_SIGN = +1  # Change to -1 to reverse right side
```

## 🚀 Usage

### Basic Operation
```bash
python3 main.py
```

### Access Video Stream
1. Find your Raspberry Pi's IP address:
   ```bash
   hostname -I
   ```

2. Open browser and navigate to:
   ```
   http://<raspberry-pi-ip>:5000
   ```

### Robot Behavior

**Normal Operation:**
1. Robot follows black line using center band detection
2. When line is lost, enters oscillating search pattern
3. After 0.8s of searching, briefly reverses and continues

**Obstacle Detected:**
1. Ultrasonic sensor triggers at <23cm
2. Robot stops and analyzes obstacle
3. YOLO model classifies color:
   - **Red** → Execute right avoidance maneuver
   - **Green** → Execute left avoidance maneuver
4. Returns to line following after avoidance

**QR Code Detected:**
1. Robot detects QR code in frame
2. Reverses briefly to clear QR area
3. Executes U-turn based on QR content
4. Spins until original line is re-centered
5. Resumes line following

## 🧠 How It Works

### Line Detection Algorithm
```
1. Capture frame from camera
2. Extract center horizontal band (15% of frame height)
3. Convert to HSV color space
4. Apply black color mask (V: 0-70)
5. Apply Gaussian blur and morphological opening
6. Divide into LEFT (20%), MIDDLE (60%), RIGHT (20%) zones
7. Find largest blob in each zone
8. Calculate steering based on MIDDLE zone centroid error
```

### Obstacle Avoidance State Machine
```
State: AVOID_RED (duration: ~3.3s)
├─ Turn right (0.6s)
├─ Drive straight (1.4s)
├─ Counter-turn left (1.4s)
└─ Final forward (0.5s)

State: AVOID_GREEN (duration: ~3.8s)
├─ Turn left (1.2s)
├─ Drive straight (1.4s)
├─ Counter-turn right (1.3s)
└─ Final forward (0.5s)
```

### U-Turn Logic
```
State: REVERSE_LEFT/RIGHT
├─ Reverse (0.5s) - clear QR area
├─ Blind spin (1.4s) - turn away from current line
├─ Visual search - spin until line re-centered
└─ Resume line following
```

## 🐛 Troubleshooting

### Camera Issues
```bash
# Test camera
raspistill -o test.jpg

# Check camera is detected
vcgencmd get_camera
```

### Motor Direction Problems
- Adjust `LEFT_SIGN` and `RIGHT_SIGN` values
- Check motor controller connections
- Verify power supply is adequate

### YOLO Model Not Loading
```bash
# Verify model path
ls -la /path/to/red-green-box.onnx

# Check ultralytics installation
pip3 show ultralytics

# Disable YOLO to use HSV fallback
YOLO_ENABLED = False
```

### Ultrasonic Sensor Unreliable
- Check GPIO connections (TRIG=21, ECHO=20)
- Verify 5V power supply
- Ensure sensor is mounted stably
- Adjust `ULTRA_CONSECUTIVE` for more reliable readings

### Robot Doesn't Follow Line
- Check lighting conditions (avoid shadows)
- Adjust `upper_black` threshold in HSV mask
- Verify camera orientation (`ROTATE_90_CW`)
- Tune `BAND_HEIGHT_FRAC` (try 0.10-0.25)

## 📊 Performance Tips

1. **Lighting**: Consistent, diffused lighting works best
2. **Line Width**: 2-3cm black tape optimal
3. **Surface**: Matte surfaces reduce glare
4. **Speed**: Lower `BASE` speed for tighter turns
5. **Camera Height**: 15-20cm above ground recommended

## 🔒 Safety Features

- **Auto-safe mode** enabled on motor controller
- **Emergency stop** via Ctrl+C
- **GPIO cleanup** on shutdown
- **Motor speed clamping** (-99 to +99 PWM)
- **Ultrasonic timeout** prevents infinite loops

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- YOLO model training using Ultralytics framework
- AUPPBot motor controller library
- OpenCV and pyzbar communities

## 👥 Contributors

- Group 6
