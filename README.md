# LIMB2-DVA513

Student project for MDU's LIMB rehabilitation system.

The main application is in `Camera_testing`. It uses an OAK-D camera, DepthAI, OpenCV and MediaPipe Tasks to record arm/hand movements, save patient JSON files, identify users in live mode, and select GA movement profiles.

## Repository Contents

Commit these:

- `Camera_testing/limb_vision_test.cpp` - main C++ application.
- `Camera_testing/BUILD` - Bazel target used from inside the MediaPipe workspace.
- `Camera_testing/nop/` - small header dependency used by DepthAI serialization.
- `Camera_testing/*.task` - MediaPipe pose/hand models used by the app.
- `Camera_testing/json_files/` - shared patient JSON data for the group.
- `Camera_testing/json_files/ga_profiles/` - optional GA output profiles per movement.

Do not commit these:

- `~/mediapipe` source tree.
- `depthai-core` source/build folders.
- Bazel output folders such as `bazel-*`.
- CMake/build output folders.
- `.DS_Store`, `.vscode`, local IDE files.
- Old local experiments in `camera_vision/`.

## Expected Folder Layout

This project is meant to live inside a local MediaPipe checkout:

```text
~/mediapipe/
  mediapipe/
  WORKSPACE
  LIMB2-DVA513/
    Camera_testing/
      limb_vision_test.cpp
      BUILD
      pose_landmarker_lite.task
      pose_landmarker_full.task
      hand_landmarker.task
      json_files/
```

The Bazel target depends on MediaPipe targets such as `//mediapipe/tasks/...`, so building from a standalone folder will not work. Clone it into the MediaPipe root.

## 1. Install System Tools

macOS setup:

```bash
xcode-select --install
brew install bazelisk cmake opencv git
```

DepthAI is also required. Each developer should install/build it locally; do not commit DepthAI itself to this repo.

One working approach is to build DepthAI Core from source and remember the build folder path:

```bash
git clone https://github.com/luxonis/depthai-core.git ~/depthai-core
cd ~/depthai-core
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

If your DepthAI library ends up somewhere else, use that path in the build command below.

## 2. Clone MediaPipe And This Project

```bash
git clone https://github.com/google-ai-edge/mediapipe.git ~/mediapipe
cd ~/mediapipe
git checkout v0.10.35
git clone https://github.com/isabellam2004/LIMB2-DVA513.git
```

If you already have `~/mediapipe`, clone only this repo into that folder:

```bash
cd ~/mediapipe
git clone https://github.com/isabellam2004/LIMB2-DVA513.git
```

## 3. Build

Set the path to your local DepthAI build:

```bash
export DEPTHAI_LIB_DIR="$HOME/depthai-core/build"
```

Build from the MediaPipe root:

```bash
cd ~/mediapipe
bazel build \
  --enable_bzlmod=false \
  --enable_workspace=true \
  --config=darwin_arm64 \
  --define MEDIAPIPE_DISABLE_GPU=1 \
  -c opt \
  --linkopt=-L"$DEPTHAI_LIB_DIR" \
  --linkopt=-Wl,-rpath,"$DEPTHAI_LIB_DIR" \
  --linkopt=-Wl,-rpath,"$DEPTHAI_LIB_DIR/_deps/dynamic_calibration-src/lib" \
  --linkopt=-Wl,-rpath,"$DEPTHAI_LIB_DIR/vcpkg_installed/arm64-osx/lib" \
  //LIMB2-DVA513/Camera_testing:limb_vision_test
```

If Bazel cannot find `depthai/depthai.hpp`, add the include path:

```bash
--cxxopt=-I"$HOME/depthai-core/include"
```

Do not run `bazel clean` unless you really need to. It can make the next build take a long time.

## 4. Run

Run from the MediaPipe root so model and JSON paths resolve correctly:

```bash
cd ~/mediapipe
./bazel-bin/LIMB2-DVA513/Camera_testing/limb_vision_test
```

Menu options:

- `1` - Training mode. Records movement JSON files into `Camera_testing/json_files`.
- `2` - Live mode. Loads patient JSON files from `Camera_testing/json_files`, identifies the patient, and checks GA movement profiles.

Press `q` in the OpenCV window to exit.

## JSON Data

Patient movement recordings are intentionally stored in:

```text
Camera_testing/json_files/
```

GA output profiles should be stored in:

```text
Camera_testing/json_files/ga_profiles/
```

The live application reads from these folders automatically.

If the JSON files contain real patient data, confirm that the group is allowed to share them in GitHub before pushing.

## Useful Git Commands

Check what will be uploaded:

```bash
git status --short
```

Add the intended project files:

```bash
git add .gitignore README.md Camera_testing
git status --short
```

Commit and push:

```bash
git commit -m "Add LIMB camera tracking application"
git push origin main
```

If the repository branch is named `master`, use:

```bash
git push origin master
```

## Troubleshooting

`Library not loaded: @rpath/libdepthai-core.dylib`

The DepthAI runtime library is not on the runtime path. Rebuild with the `--linkopt=-Wl,-rpath,...` lines above and make sure `DEPTHAI_LIB_DIR` points to the folder containing `libdepthai-core.dylib`.

`fatal error: 'mediapipe/...' file not found`

Build from inside `~/mediapipe`, not from inside `LIMB2-DVA513`.

`fatal error: 'depthai/depthai.hpp' file not found`

DepthAI headers are missing from the compiler include path. Install/build DepthAI and pass `--cxxopt=-I/path/to/depthai-core/include`.

Build takes many minutes

Avoid changing Bazel flags between builds and avoid `bazel clean --expunge`. After the first successful build, normal C++ edits should rebuild much faster.
