# Rakib_FYP_XAI Audio Recorder App

A cross-platform mobile audio recording application built with Python, Kivy, and KivyMD.

## Features

- Beautiful and lightweight UI
- Audio recording with microphone access
- Audio playback with pause capability
- Real-time waveform visualization
- Save recordings in MP3 or WAV format
- Open and play existing audio files

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   python main.py
   ```

## Requirements

- Python 3.7+
- Kivy 2.2.1+
- KivyMD 1.1.1+
- PyAudio 0.2.13+
- Pydub 0.25.1+
- NumPy 1.24.3+
- Matplotlib 3.7.1+

## Building for Mobile

### Android

1. Install Buildozer:
   ```
   pip install buildozer
   ```
2. Initialize Buildozer:
   ```
   buildozer init
   ```
3. Edit the buildozer.spec file to include the required dependencies
4. Build the APK:
   ```
   buildozer android debug
   ```

### iOS

For iOS, you'll need to use Kivy's toolchain. Please refer to the official Kivy documentation for detailed instructions.

## Building for Android (Alternative Approaches)

For detailed build instructions, see [Quick Build Guide](quick_build_guide.md).

### Easiest Approach

Run the all-in-one setup and build script:

```bash
./setup_and_build.sh
```

This script handles:
- Installing Java 17 if needed
- Setting up Python dependencies
- Creating a minimal configuration
- Building a test APK

### Alternative Build Methods

Several specialized build scripts are available:
- `build_apk_only.sh`: Standard build with full features
- `build_minimal_apk.sh`: Simplified build with fewer dependencies
- `build_with_beeware.sh`: Uses BeeWare's Briefcase instead of Buildozer
- `online_build.sh`: Prepares files for building on Google Colab

## Development

The application is built with Kivy and uses the following libraries:
- PyAudio for sound recording
- Matplotlib for waveform visualization
- KivyMD for Material Design UI components

## License

This project is licensed under the MIT License - see the LICENSE file for details.
