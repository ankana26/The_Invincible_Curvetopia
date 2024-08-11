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

**2. Exploring Symmetry in Curves**
For closed shapes, the task is to identify the presence of symmetry, focusing initially on reflection symmetries. This involves:
  - Symmetry Hunt: Detecting lines of symmetry where a shape can be divided into mirrored halves.
  - Bezier Curve Fitting: Fitting Bezier curves on symmetric points to ensure a consistent representation.

**3. Completing Incomplete Curves**
The aim is to complete incomplete 2D curves that have gaps or partial holes due to planarization, where overlapping portions have been removed. The goal is to use computer vision techniques to naturally complete these curves, taking into account various levels of shape occlusion and ensuring smoothness, regularity, and symmetry in the completed curves.

## Scripts
The script files are mentioned below:<br>

**Dependencies**
- numpy
- matplotlib
- scipy
- sklearn
- opencv-python

**1. code1_line_seg.py**
  - Description: This script detects line segments in 2D curves using the Hough Line Transform. It reads curves from a CSV file, processes them to detect line segments, and plots both the original curves and the detected line segments.
    
- Usage:
  - you can run the python file using: python code1_line_seg.py
  - Make sure to replace csv_path with the path to your CSV file containing the curve data.
    
- Parameters:
  - rho: Distance resolution of the accumulator in pixels.
  - theta: Angle resolution of the accumulator in radians.
  - threshold: Threshold for line detection.
  - min_line_length: Minimum length of line segments.
  - max_line_gap: Maximum allowed gap between line segments.

**2. code2_circle.py**
- Description: This script detects circles and ellipses from 2D curves using clustering and geometric fitting. It reads curves from a CSV file, fits circles and ellipses to the clustered points, and plots the detected shapes.
 
- Usage: 
  - you can run the python file using: python code2_circle.py
  - Make sure to replace csv_path with the path to your CSV file.
 
- Parameters:
  - min_points: Minimum number of points required to fit a shape.
  - max_distance: Maximum distance between points for clustering.
  - min_samples: Minimum number of samples for clustering.
  - circle_threshold: Threshold for fitting circles.
  - match_ratio: To detect regular circles in image, keep the parameter value as 0.2 else for the occluded circles the value should be 0.1
  - ellipse_threshold: Threshold for fitting ellipses.

**3. code3_rect.py**
- Description: This script detects star-shaped patterns in 2D curves by approximating them with Bezier curves and analyzing their curvature. It also uses DBSCAN for clustering.
  
- Usage:
  - you can run the python file using: python code3_rect.py
  - Make sure to replace csv_path with the path to your CSV file.

- Parameters:
  - eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other (DBSCAN parameter). This value should be 4 optimally, increase this value if less shapes needed to be detceted while decrease this value if more shapes needed to be detecetd.
  - min_samples: The number of samples in a neighborhood for a point to be considered as a core point (DBSCAN parameter).
  - error_threshold: Maximum allowed error for fitting the Bezier curve to detect star shapes.

**4. code4_star.py**
-Description: This script detects star-shaped patterns in 2D curves by approximating them with Bezier curves and analyzing their curvature. It also utilizes DBSCAN clustering to group points.

- Usage
  - To run the script, execute the following command: python code3_rect.py
  - Ensure you replace csv_path with the path to your CSV file.

- Parameters
  - eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other (DBSCAN parameter). This value should be 4 optimally, increase this value if less shapes needed to be detceted while decrease this value if more shapes needed to be detecetd.
  - min_samples: The number of samples in a neighborhood for a point to be considered as a core point (DBSCAN parameter).
  - error_threshold: Maximum allowed error for fitting the Bezier curve to detect star shapes.

**5. code5_symmetry.py**
-Description: This script detects symmetry axes in 2D curves by clustering points and evaluating potential symmetry lines. It utilizes DBSCAN for clustering and optimizes the maximum distance for symmetry evaluation.
  
- Usage
  - To run the script, execute the following command: python code5_symmetry.py
  - Ensure you replace csv_path with the path to your CSV file.

- Parameters
  - eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other (DBSCAN parameter). This value should be 4 optimally, increase this value if less shapes needed to be detceted while decrease this value if more shapes needed to be detecetd.
  - min_samples: The number of samples in a neighborhood for a point to be considered as a core point (DBSCAN parameter).
