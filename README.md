# RealSense D435i Vision Pipeline for Tube Detection

This repository contains a Python-based computer vision pipeline for an Intel RealSense D435i camera.

The current goal is to detect colored cosmetic tubes, identify their color, estimate a stable 3D point in camera coordinates, and move toward a future vision-guided pick-and-place demo.

This repo is focused on the **camera / vision side only**.
Robot motion, grasp execution, and hand-eye calibration are not implemented here.

## Project Scope

Hardware and environment:

- Intel RealSense D435i
- Windows laptop
- Python 3.9
- `pyrealsense2`
- `opencv-python`
- Camera currently tested both on-table and in a future eye-in-hand setup

Robot context:

- Future target system: Neuromeka Indy7 + gripper
- Real demo setup: RealSense D435i mounted on the robot wrist, pointing downward to the pick area
- This repository does **not** control the robot

Objects:

- Cosmetic facewash tubes
- Main colors:
  - Blue
  - Yellow
  - Green
- Dark green is treated as part of the `GREEN` class

## Current Vision Goals

The current pipeline is being developed in stages:

1. Detect one tube as one object
2. Label the dominant color
3. Estimate object orientation angle in the image
4. Estimate a stable 3D point in camera coordinates
5. Estimate a cap-side grasp point for future pick-and-place

For the flat-tube demo, the intended final output is:

- `label`
- `angle_deg`
- `grasp_px`, `grasp_py`
- `grasp_X_m`, `grasp_Y_m`, `grasp_Z_m`
- stable target selection

## Repository Structure

Main files:

- [testing/live_view.py](testing/live_view.py)
  - Basic RealSense color + depth viewer
  - Click a pixel to inspect depth in meters

- [testing/test_camera_detect.py](testing/test_camera_detect.py)
  - Quick check that Python can see the RealSense device

- [testing/color_detect.py](testing/color_detect.py)
  - Early color-mask detector for blue / green / yellow objects

- [testing/object_color_detect.py](testing/object_color_detect.py)
  - Upright-object pipeline
  - Designed around one vertical object becoming one box with center `X, Y, Z`
  - Useful for early debugging and color/depth tuning

- [testing/flat_tube_detect.py](testing/flat_tube_detect.py)
  - Current main script for flat tubes lying on a table
  - Detects flat tubes using color + depth + contour geometry
  - Estimates angle and a cap-side grasp point

## Current Main Script

Use this script for the flat-table tube demo:

```bash
python testing/flat_tube_detect.py
```

The script currently does the following:

- Starts RealSense color and depth streams
- Aligns depth to the color image
- Converts the color frame to HSV
- Builds blue / green / yellow masks
- Combines the color mask with a near-depth mask
- Finds contours from the foreground mask
- Builds raw candidates from tube-like contours
- Merges split same-color pieces when they likely belong to one tube
- Computes a rotated box and tube angle
- Estimates a cap-side grasp point from tube geometry
- Deprojects the grasp pixel into `X, Y, Z` camera coordinates
- Chooses a nearest valid target and keeps it stable across frames

Displayed windows:

- `Color`
- `Combined Mask`
- `Near Mask`
- `Foreground Mask`
- `Depth`

Terminal output includes:

- color label
- center pixel
- grasp pixel
- `GX`, `GY`, `GZ`
- `angle`
- `part_count`
- cap confidence

## Detection Logic Summary

### Color

The pipeline uses HSV color thresholds to detect:

- `BLUE`
- `GREEN`
- `YELLOW`

### Depth

A near-depth mask is used to keep only objects within a chosen working range.

### Shape

For flat tubes, the script uses:

- contour area filtering
- aspect ratio filtering
- rotated rectangle geometry
- split-piece merging for same-color fragments

### Cap-side grasp point

The current flat-tube script estimates the cap side from geometry:

- tube width is analyzed along the long axis
- the cap end is assumed to be the narrower end
- the grasp point is placed slightly inward from the cap tip

This is still a development-stage heuristic, not a production grasp detector.

## Recommended Test Setup

The pipeline works best when the physical setup is controlled.

Recommended setup:

- Keep the camera fixed, not hand-held
- Use one fixed working height
- Use a matte background if possible
- Keep the whole tube visible with some empty border around it
- Avoid placing the camera too near the object
- Avoid strong glare on shiny tubes, especially yellow

Why:

- Glossy reflections can make colored regions appear white
- White reflections can break the color mask
- Close-range viewing increases glare, distortion, and depth instability
- Yellow is especially sensitive to this problem

## Practical Testing Order

Recommended test sequence:

1. One yellow tube only
2. One blue tube only
3. One green tube only
4. Two separated tubes
5. Three separated tubes
6. Wrist-mounted fixed-pose camera test

For each single-tube test:

- test several orientations
- keep the scene still for a few seconds
- watch whether:
  - one final tube is detected
  - the angle follows the real tube
  - the grasp point stays on the cap side
  - the target remains stable

## Current Limitations

This project is still experimental.

Known limitations:

- Detection is sensitive to shiny surfaces and lighting changes
- Yellow is more sensitive to glare than blue
- Very close camera distance can reduce stability
- Touching or overlapping tubes are harder to handle
- Cap-side estimation is heuristic and may fail in difficult views
- No robot-side integration is included yet
- No camera-to-robot calibration is included yet

## Planned Next Steps

Near-term vision milestones:

1. Stabilize the flat-tube detector at the real wrist-mounted inspection pose
2. Validate grasp point stability from the real robot camera view
3. Tune color and geometry thresholds for the final demo background and lighting
4. Prepare the vision output for robot-side integration

Future robot-related steps, outside this repo for now:

- hand-eye calibration
- transform camera coordinates to robot coordinates
- motion planning
- pick and place execution

## Installation

Install dependencies in your Python environment:

```bash
pip install pyrealsense2 opencv-python numpy
```

Notes:

- This repository is mainly being developed on Windows with Python 3.9
- The D435i has also been tested under USB 2.x settings:
  - depth: `640x480 @ 15 fps`
  - color: `640x480 @ 30 fps`

## Quick Start

Check that Python can see the camera:

```bash
python testing/test_camera_detect.py
```

Basic live view:

```bash
python testing/live_view.py
```

Basic color detection:

```bash
python testing/color_detect.py
```

Upright-object pipeline:

```bash
python testing/object_color_detect.py
```

Flat-tube pipeline:

```bash
python testing/flat_tube_detect.py
```

## Notes

This repo is intentionally narrow in scope.

It is a working vision sandbox for:

- RealSense image acquisition
- color mask tuning
- tube detection
- 3D point estimation
- cap-side grasp-point estimation

It is not yet a complete robot pick-and-place system.
