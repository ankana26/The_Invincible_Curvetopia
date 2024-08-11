import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
import os

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

def bernstein_poly(i, n, t):
    return comb(n, i) * (t**(n-i)) * (1 - t)**i

def bezier_curve(points, num=200):
    n = len(points) - 1
    t = np.linspace(0, 1, num)
    curve = np.zeros((num, 2))
    for i, point in enumerate(points):
        curve += np.outer(bernstein_poly(i, n, t), point)
    return curve

def compute_curvature(curve):
    dx_dt = np.gradient(curve[:, 0])
    dy_dt = np.gradient(curve[:, 1])
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)
    curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt)**1.5
    return curvature

def bezier_curve_error(points, curve):
    errors = []
    for point in points:
        distances = np.sum((curve - point)**2, axis=1)
        min_distance = np.min(distances)
        errors.append(min_distance)

    return np.mean(errors)
def is_star_shape(points, num_peaks=5, curvature_threshold=0.1, error_threshold=1.0):
    curve = bezier_curve(points)
    error = bezier_curve_error(points, curve)
    
    if error > error_threshold:
        return False
    
    curvature = compute_curvature(curve)
    
    # Find peaks in curvature
    peaks = np.where((curvature[1:-1] > curvature[:-2]) & 
                     (curvature[1:-1] > curvature[2:]) & 
                     (curvature[1:-1] > curvature_threshold))[0] + 1
    
    return len(peaks) >= num_peaks

def detect_stars(paths_XYs, eps=4, min_samples=5, min_points=20, error_threshold=1.0):
    all_points = np.vstack([xy for XYs in paths_XYs for XY in XYs for xy in XY])
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(all_points)
    
    clusters = []
    detected_stars = []
    
    for label in set(labels):
        if label == -1:  # Noise points
            continue
        
        cluster_points = all_points[labels == label]
        clusters.append(cluster_points)
        
        if len(cluster_points) < min_points:
            continue
        
        if is_star_shape(cluster_points, error_threshold=error_threshold):
            detected_stars.append(cluster_points)
    
    return clusters, detected_stars

def plot(paths_XYs, clusters=None, detected_stars=None):
    fig, ax = plt.subplots(tight_layout=True, figsize=(12, 12))
    colours = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    
    # Plot original curves
    for i, XYs in enumerate(paths_XYs):
        c = colours[i % len(colours)]
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2, alpha=0.5)
    
    # Plot clusters
    if clusters is not None:
        for i, cluster in enumerate(clusters):
            ax.scatter(cluster[:, 0], cluster[:, 1], c=colours[i % len(colours)], s=20, alpha=0.7)
    
    # Plot detected stars
    if detected_stars:
        for star in detected_stars:
            curve = bezier_curve(star)
            ax.plot(curve[:, 0], curve[:, 1], 'k-', lw=2)
    
    ax.set_aspect('equal')
    plt.show()

def save_detected_shapes_to_csv(detected_stars, original_file):
    # Create the new filename
    base_name = os.path.basename(original_file)
    name_without_ext = os.path.splitext(base_name)[0]
    new_filename = f"{name_without_ext}_sol.csv"
    new_filepath = os.path.join(os.path.dirname(original_file), new_filename)

    # Prepare the data for saving
    all_points = []
    for i, star in enumerate(detected_stars):
        for j, point in enumerate(star):
            all_points.append([i, j, point[0], point[1]])

    # Save to CSV
    np.savetxt(new_filepath, all_points, delimiter=',', fmt='%d,%d,%.6f,%.6f')
    print(f"Detected shapes saved to {new_filepath}")

# Main execution
csv_path = "problems/isolated.csv"
paths_XYs = read_csv(csv_path)

# Detect stars
error_threshold = 10  # Adjust this value as needed
clusters, detected_stars = detect_stars(paths_XYs, error_threshold=error_threshold)

print(f"Number of clusters detected: {len(clusters)}")
print(f"Number of stars detected: {len(detected_stars)}")

save_detected_shapes_to_csv(detected_stars, csv_path)
# Plot the results
plot(paths_XYs, clusters, detected_stars)