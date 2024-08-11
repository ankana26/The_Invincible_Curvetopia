# Curve Regularization and Symmetry Detection in 2D Euclidean Space
## Objective
This repository contains Python scripts for detecting various geometric shapes and patterns from 2D curve data. The scripts use different algorithms and methods to identify line segments, circles, ellipses, star shapes, and symmetry in the given datasets. 

## Key Sections
**1. Regularize Curves:**
The goal is to identify regular shapes among a set of curves. This task is broken down into the following primitives:

  - Straight Lines: Detect straight line segments in the input paths.
  - Circles and Ellipses: Recognize curves where all points are equidistant from a center (circle) or have two focal points (ellipse).
  - Rectangles and Rounded Rectangles: Identify rectangles and distinguish between rectangles with sharp and rounded edges.
  - Regular Polygons: Detect polygons with equal sides and angles.
  - Star Shapes: Identify star shapes based on a central point with multiple radial arms.
  This activity primarily targets hand-drawn shapes and doodles. The algorithm should also be capable of distinguishing between regular and non-regular shapes.

## Scripts
The script files are mentioned below:
**1. code1_line_seg.py**
  - Description: This script detects line segments in 2D curves using the Hough Line Transform. It reads curves from a CSV file, processes them to detect line segments, and plots both the original curves and the detected line segments.

- Dependencies:
  - numpy
  - matplotlib
  - opencv-python
    
- Usage:
  - you can run the python file using: python code1_line_seg.py
  - Make sure to replace csv_path with the path to your CSV file containing the curve data.
    
- Parameters:
  - rho: Distance resolution of the accumulator in pixels.
  - theta: Angle resolution of the accumulator in radians.
  - threshold: Threshold for line detection.
  - min_line_length: Minimum length of line segments.
  - max_line_gap: Maximum allowed gap between line segments.
