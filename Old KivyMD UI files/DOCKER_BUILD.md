# Building XAIPhonogram APK with Docker

This guide explains how to build the XAIPhonogram Android APK using Docker, which provides a consistent environment for compilation regardless of your host operating system.

## Prerequisites

- Docker installed on your machine
  - macOS: Install [Docker Desktop for Mac](https://docs.docker.com/desktop/install/mac-install/)
  - Windows: Install [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/)
  - Linux: Install [Docker Engine](https://docs.docker.com/engine/install/)
- The XAIPhonogram source code (this repository)

## Building the APK

### One-Step Build

The simplest way to build is to use the provided script:

```bash
# Make sure the script is executable
chmod +x build_apk_docker.sh

# Run the build script
./build_apk_docker.sh
```

This script will:
1. Build a Docker image with all necessary dependencies
2. Mount your current directory into the Docker container
3. Run buildozer inside the container to create the APK
4. Output the APK to the `bin/` directory

### Manual Steps

If you prefer to run the commands manually:

1. Build the Docker image:
   ```bash
   docker build -t xaiphonogram-builder:latest .
   ```

2. Copy the buildozer spec file:
   ```bash
   cp buildozer.simple.spec buildozer.spec
   ```

3. Run the build process:
   ```bash
   docker run --rm -v "$(pwd)":/home/user/app xaiphonogram-builder:latest buildozer android debug
   ```

## Troubleshooting

### Common Issues

1. **Docker permission issues**: 
   - On Linux, you might need to run Docker commands with `sudo` or add your user to the docker group

2. **Build fails with Java-related errors**:
   - The Dockerfile uses OpenJDK 11. If you need a different version, modify the Dockerfile

3. **Buildozer errors**:
   - Check the logs in the `logs/` directory for detailed error messages
   - Try running `docker run --rm -v "$(pwd)":/home/user/app xaiphonogram-builder:latest buildozer android clean` before building again

### Getting Debug Information

To start an interactive session in the Docker container:

```bash
docker run --rm -it -v "$(pwd)":/home/user/app xaiphonogram-builder:latest bash
```

## Installing the APK

After a successful build, the APK will be in the `bin/` directory. You can install it on an Android device using:

```bash
adb install -r bin/xaiphonogram-*-debug.apk
```

## Notes on Performance

The first build will take a considerable amount of time (15-30 minutes) as it needs to download Android SDK components and dependencies. Subsequent builds will be faster.
