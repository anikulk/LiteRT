# LiteRT + OpenVINO (Android x86_64) Build Notes

This document summarizes the meaningful steps for building LiteRT with the Intel OpenVINO vendor plugin and applying the plugin to a TFLite model.

## 1) OpenVINO runtime location

###  OpenVINO nightly bundle

```bash
cd ~/workspace/ov-rel
wget https://storage.openvinotoolkit.org/repositories/openvino/packages/nightly/2026.1.0-20996-3ef8425ae12/openvino_toolkit_ubuntu22_2026.1.0.dev20260131_x86_64.tgz
 tar -xvf openvino_toolkit_ubuntu22_2026.1.0.dev20260131_x86_64.tgz
```

## 2) Clone and set up LiteRT

```bash
export REBASE_DIR=/your/working/directory
git clone https://github.com/google-ai-edge/LiteRT $REBASE_DIR/litert
cd $REBASE_DIR/litert
./configure
```

## 3) Build LiteRT + OpenVINO vendor libraries (Linux) 

```bash
cd $REBASE_DIR/litert
./configure

bazel-7.4.1-linux-x86_64 build -c opt 
  --per_file_copt=.*\.cpp@-fexceptions \
  --per_file_copt=.*\.cc@-fexceptions \
  --cxxopt=-frtti \
  --linkopt="-Wl,-z,max-page-size=16384" \
  litert/vendors/intel_openvino/compiler:libLiteRtCompilerPlugin_IntelOpenvino.so

bazel-7.4.1-linux-x86_64 build -c opt 
  --per_file_copt=.*\.cpp@-fexceptions \
  --per_file_copt=.*\.cc@-fexceptions \
  --cxxopt=-frtti \
  --linkopt="-Wl,-z,max-page-size=16384" \
  litert/vendors/intel_openvino/dispatch:libLiteRtDispatch_IntelOpenvino.so

bazel-7.4.1-linux-x86_64 build -c opt 
  --linkopt="-Wl,-z,max-page-size=16384" \
  litert/c:libLiteRt.so
```

## 4) Build tools (apply_plugin / benchmark_model) Linux

```bash
bazel-7.4.1-linux-x86_64 build -c opt  \
  --linkopt="-Wl,-z,max-page-size=16384" \
  litert/tools:apply_plugin_main

bazel-7.4.1-linux-x86_64 build -c opt  \
  --linkopt="-Wl,-z,max-page-size=16384" \
  litert/tools:benchmark_model
```


## 5) Android NDK/SDK setup (optional)

```bash
export OPV_HOME_DIR=~/workspace/openvino-android

# Android NDK
wget https://dl.google.com/android/repository/android-ndk-r29-linux.zip --directory-prefix $OPV_HOME_DIR
unzip $OPV_HOME_DIR/android-ndk-r29-linux.zip -d $OPV_HOME_DIR
mv $OPV_HOME_DIR/android-ndk-r29 $OPV_HOME_DIR/android-ndk
export ANDROID_NDK_PATH=$OPV_HOME_DIR/android-ndk

# Android SDK tools
wget https://dl.google.com/android/repository/platform-tools-latest-linux.zip --directory-prefix $OPV_HOME_DIR
unzip $OPV_HOME_DIR/platform-tools-latest-linux.zip -d $OPV_HOME_DIR/android-sdk
export ANDROID_TOOLS_PATH=$OPV_HOME_DIR/android-sdk/platform-tools

wget https://dl.google.com/android/repository/commandlinetools-linux-13114758_latest.zip --directory-prefix $OPV_HOME_DIR
unzip $OPV_HOME_DIR/commandlinetools-linux-13114758_latest.zip -d $OPV_HOME_DIR/android-sdk/

# JDK
sudo apt install openjdk-17-jdk

# Fix cmdline-tools layout
cd $OPV_HOME_DIR/android-sdk/cmdline-tools
mkdir latest
mv $OPV_HOME_DIR/android-sdk/cmdline-tools/bin/ latest/
mv $OPV_HOME_DIR/android-sdk/cmdline-tools/lib/ latest/
mv $OPV_HOME_DIR/android-sdk/cmdline-tools/NOTICE.txt latest/
mv $OPV_HOME_DIR/android-sdk/cmdline-tools/source.properties latest/

# Install SDK packages
$OPV_HOME_DIR/android-sdk/cmdline-tools/latest/bin/sdkmanager "build-tools;35.0.0"
$OPV_HOME_DIR/android-sdk/cmdline-tools/latest/bin/sdkmanager "build-tools;36.0.0"
$OPV_HOME_DIR/android-sdk/cmdline-tools/latest/bin/sdkmanager "platform-tools" "platforms;android-36" "platforms;android-35" "platforms;android-34"
```

## 6) Android build environment variables (optional)

```bash
export CURRENT_ANDROID_ABI=x86_64
export CURRENT_ANDROID_PLATFORM=android-35
export CURRENT_ANDROID_STL=c++_shared
export CURRENT_CMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_PATH/build/cmake/android.toolchain.cmake
export CURRENT_ANDROID_TOOLCHAIN_ROOT=$ANDROID_NDK_PATH/toolchains/llvm/prebuilt/linux-x86_64
```


## 7) Apply the OpenVINO plugin to a TFLite model

```bash
# Use OpenVINO runtime libs in LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$OPENVINO_NATIVE_DIR/runtime/lib/intel64

./bazel-bin/litert/tools/apply_plugin_main --cmd apply \
  --model=/model/to/test \
  -o aot_model.tflite \
  --libs bazel-bin/litert/vendors/intel_openvino/compiler/ \
  --intel_openvino_device_type=cpu \
  --soc_manufacturer "IntelOpenVINO"
```

## Notes

- `apply_plugin_main` expects the OpenVINO runtime libraries in `LD_LIBRARY_PATH`.
- Android builds use the `android_x86_64` Bazel config and `litert_android_no_jni=true`.
- The `--intel_openvino_configs_map` was frequently set to `NPU_COMPILER_TYPE=PLUGIN,NPU_PLATFORM=5010`. for npu
