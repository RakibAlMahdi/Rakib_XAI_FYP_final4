# Building XAIPhonogram APK in Google Colab - Troubleshooting Guide

## The Problem

When building the Android APK using Google Colab, you may encounter an error related to libffi compilation, specifically with autoconf macros:

```
configure.ac:41: error: possibly undefined macro: AC_PROG_LIBTOOL
configure:8578: error: possibly undefined macro: AC_PROG_LD
autoreconf: error: /usr/bin/autoconf failed with exit status: 1
```

This is a common issue caused by missing build dependencies in the Colab environment.

## Solution

We've prepared two files to help you overcome this issue:

1. **buildozer.colab.spec**: A simplified specification file optimized for Colab builds
2. **colab_build_fix.sh**: A shell script that installs missing dependencies and patches the libffi build system

## How to Use This Fix in Colab

1. Upload your project files to Colab, including both solution files
2. Run the following commands in a Colab cell:

```python
# Install buildozer
!pip install buildozer cython

# Install Android SDK tools
!apt-get update
!apt-get install -y git zip unzip openjdk-8-jdk python3-pip autoconf libtool pkg-config zlib1g-dev libncurses5-dev libncursesw5-dev libtinfo5 cmake

# Set environment variables
import os
os.environ['PATH'] = "/usr/local/bin:" + os.environ['PATH']

# Make the fix script executable
!chmod +x colab_build_fix.sh

# Copy the Colab-specific buildozer spec
!cp buildozer.colab.spec buildozer.spec

# Run the fix script
!bash colab_build_fix.sh
```

## Alternative Approach

If the above solution doesn't work, you can try a more direct approach:

```python
# Install dependencies
!apt-get update
!apt-get install -y libtool automake autoconf pkg-config build-essential libtool-bin

# Install buildozer and its dependencies
!pip install buildozer cython

# Install Android build tools
!apt-get install -y git zip unzip openjdk-8-jdk

# Create a simple requirements file
!echo "python3,kivy==2.1.0,kivymd,pillow" > requirements.txt

# Use p4a directly with workarounds
!pip install python-for-android
!python -m pythonforandroid.toolchain create --dist_name=xaiphonogram --bootstrap=sdl2 --requirements=@requirements.txt --arch=arm64-v8a --copy-libs --ignore-setup-py
```

## Retrieving Your APK

After a successful build, the APK will be in the `/content/bin/` directory. You can download it directly from the Colab file browser or use this command to get a download link:

```python
from google.colab import files
files.download('/content/bin/xaiphonogram-1.0.0-arm64-v8a-debug.apk')
```

## Common Issues

1. **Memory/resources exhausted**: Colab sessions have limited resources. Try reducing parallel builds with `--debug` and `--color=always` flags.

2. **SDK/NDK installation failing**: Use a specific version with `--android-api 29` and `--ndk-api 21`.

3. **libffi errors persisting**: Try using `--ignore-setup-py` with p4a or buildozer.

4. **Permission errors**: Add `android.accept_sdk_license = True` to your spec file.

## Need More Help?

If you continue to encounter issues, try one of these options:

1. Visit the Python-for-Android documentation: https://python-for-android.readthedocs.io/
2. Check the Buildozer issues page: https://github.com/kivy/buildozer/issues
3. Look at the p4a repository: https://github.com/kivy/python-for-android
