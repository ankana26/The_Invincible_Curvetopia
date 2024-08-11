import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter

def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs

def plot(paths_XYs, detected_shapes=None):
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    colours = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for i, XYs in enumerate(paths_XYs):
        c = colours[i % len(colours)]
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
    
    if detected_shapes:
        for shape, contour in detected_shapes:
            if shape == 'rectangle':
                # Draw the detected rectangle directly from contour
                contour = np.array(contour).reshape(-1, 2)
                ax.plot(contour[:, 0], contour[:, 1], 'y-', linewidth=2)
            elif shape == 'circle':
                (x, y), radius = cv2.minEnclosingCircle(contour)
                circle = plt.Circle((x, y), radius, color='y', fill=False, linewidth=2)
                ax.add_patch(circle)
            elif shape == 'ellipse':
                ellipse = cv2.fitEllipse(contour)
                ellipse_patch = patches.Ellipse((ellipse[0][0], ellipse[0][1]), 
                                                ellipse[1][0], ellipse[1][1], 
                                                angle=ellipse[2], color='y', 
                                                fill=False, linewidth=2)
                ax.add_patch(ellipse_patch)
    
    ax.set_aspect('equal')
    plt.show()

def preprocess_curves(paths_XYs, img_size=(500, 500)):
    img = np.zeros(img_size, dtype=np.uint8)
    for XYs in paths_XYs:
        for XY in XYs:
            pts = np.array(XY, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(img, [pts], isClosed=False, color=255, thickness=3)
    return img

def angle_between(v1, v2):
    dot = np.dot(v1, v2)
    det = v1[0]*v2[1] - v1[1]*v2[0]
    angle = np.arctan2(det, dot)
    return np.abs(angle)

def is_rectangle(contour, angle_threshold=np.pi/12, side_ratio_threshold=0.2):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    
    if len(approx) == 4:
        pts = approx.reshape(4, 2)
        angles = []
        side_lengths = []
        
        for i in range(4):
            v1 = pts[(i+1)%4] - pts[i]
            v2 = pts[(i-1)%4] - pts[i]
            angles.append(angle_between(v1, v2))
            side_lengths.append(np.linalg.norm(v1))
        
        # Check if all angles are close to 90 degrees
        if all(abs(angle - np.pi/2) < angle_threshold for angle in angles):
            # Check if opposite sides have similar lengths
            if (abs(side_lengths[0] - side_lengths[2]) / max(side_lengths[0], side_lengths[2]) < side_ratio_threshold and
                abs(side_lengths[1] - side_lengths[3]) / max(side_lengths[1], side_lengths[3]) < side_ratio_threshold):
                return True
    
    return False

def detect_shapes(img, min_contour_area=100):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_shapes = []
    shape_counts = Counter()
    
    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:
            if is_rectangle(contour):
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                detected_shapes.append(('rectangle', box))
                shape_counts['rectangle'] += 1
            else:
                # Check for other shapes (circle, ellipse) as in your original code
                pass
    
    # Print shape counts
    for shape, count in shape_counts.items():
        print(f"{shape.capitalize()} detected {count} times.")
    
    return detected_shapes

# Main execution
csv_path = 'problems/isolated.csv'
paths_XYs = read_csv(csv_path)

# Preprocess the curves to create a binary image
img = preprocess_curves(paths_XYs)

# Detect shapes in the binary image
detected_shapes = detect_shapes(img)

# Plot the original curves and the detected shapes
plot(paths_XYs, detected_shapes)