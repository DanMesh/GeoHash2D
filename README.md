# GeoHash2D
Tracker for 2D objects on a single plane using geometric hashing.

Developed as an alternative to the ASM-inspired method in the [EdgeTracker](https://github.com/DanMesh/EdgeTracker "EdgeTracker") repo.

## OpenCV Setup

### 1. Download OpenCV
```
brew install pkg-config
brew install opencv
```

### 2. Set XCode search paths
In the project's Build Settings:
- Set 'Header Search Paths' to `/usr/local/Cellar/opencv/3.4.2/include`
- Set 'Library Search Paths' to `/usr/local/Cellar/opencv/3.4.2/lib`

### 3. Add the relevant frameworks to the project
- Right click on the project in the naviagtor and select "Add Files to [Project]..."
- Navigate to `/usr/local/Cellar/opencv/3.4.2/lib/`
- Add the following files:
```
libopencv_core.3.4.2.dylib
libopencv_highgui.3.4.2.dylib
libopencv_imgcodecs.3.4.2.dylib
etc...
```
(choose files as per the included libraries in `main.cpp`)
