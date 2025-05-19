# Comprehensive Guide to Building XAIPhonogram APK

This guide provides multiple approaches to build the XAIPhonogram APK for Android, including both offline (local) and online methods.

## Option 1: Use the Provided Build Scripts

We have created several specialized build scripts to help with different build scenarios:

### Standard Build (Requires time and patience)
```bash
./build_apk_only.sh
```
This script uses a modified buildozer configuration designed to avoid AAB-related errors and will build a full-featured APK. Initial build will take 15-30 minutes to download dependencies.

### Minimal Build (Faster, fewer features)
```bash
./build_minimal_apk.sh
```
This builds a simplified version of the app without advanced features like audio recording, which can be helpful for testing the deployment process.

### BeeWare Briefcase Build (Alternative approach)
```bash
./build_with_beeware.sh
```
Uses BeeWare's Briefcase instead of Buildozer, which may work better on some systems.

## Option 2: Online Building with Google Colab (No local setup needed)

1. Run the preparation script:
```bash
./online_build.sh
```

2. Follow the printed instructions to build the APK using Google Colab. This is often the fastest approach as Colab provides a pre-configured environment.

## Option 3: Manual Build with Buildozer

If you prefer to manually configure the build:

1. Install dependencies
```bash
# macOS
brew install autoconf automake libtool pkg-config sdl2 sdl2_image sdl2_ttf sdl2_mixer
pip install buildozer cython==0.29.33
```

2. Adjust buildozer.spec as needed (several templates available in this project)

3. Run buildozer
```bash
buildozer android debug
```

## Troubleshooting Common Issues

### AAB Support Error
If you see: "buildozer version requires a python-for-android version with AAB support"
- Use the `buildozer.apk.spec` configuration which specifies a compatible p4a version

### OpenSSL Issues on macOS
If you encounter OpenSSL errors:
- Try using the Briefcase approach (`build_with_beeware.sh`) which handles dependencies differently

### Missing Android SDK or NDK
- The first build will download these automatically, but it can take time
- If the download fails, check your internet connection and try again

### Java Version Issues
If you see JDK version errors:
- Install Java 8 or 11 (both known to work well with Buildozer)
- Set JAVA_HOME environment variable to point to your Java installation

## Testing the APK

Once built, you can find the APK in the `bin/` directory. To install:

1. Transfer the APK to your Android device
2. Enable "Install from Unknown Sources" in settings
3. Tap the APK file to install

## Additional Resources

- [Buildozer Documentation](https://buildozer.readthedocs.io/)
- [BeeWare Briefcase Documentation](https://briefcase.readthedocs.io/)
- [Python-for-Android Documentation](https://python-for-android.readthedocs.io/)
