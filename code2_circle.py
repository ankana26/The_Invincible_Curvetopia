import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from sklearn.cluster import DBSCAN
from skimage import measure
import matplotlib.patches as mpatches

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

def plot(path_XYs, circles=None, ellipses=None):
    fig, ax = plt.subplots(tight_layout=True, figsize=(12, 12))
    colours = plt.cm.rainbow(np.linspace(0, 1, len(path_XYs)))

    for i, XYs in enumerate(path_XYs):
        c = colours[i]
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=1)

    if circles:
        for circle in circles:
            circle_plot = plt.Circle((circle[0], circle[1]), circle[2], fill=False, color='r', linewidth=3)
            ax.add_artist(circle_plot)

    if ellipses:
        for ellipse in ellipses:
            ellipse_plot = mpatches.Ellipse((ellipse[0], ellipse[1]), 2 * ellipse[2], 2 * ellipse[3], 
                                            angle=np.degrees(ellipse[4]), fill=False, color='g', linewidth=3)
            ax.add_artist(ellipse_plot)

    ax.set_aspect('equal')
    plt.show()

def distance_to_circle(params, points):
    xc, yc, r = params
    return np.sqrt((points[:, 0] - xc)**2 + (points[:, 1] - yc)**2) - r

def fit_circle(points):
    x_m = np.mean(points[:, 0])
    y_m = np.mean(points[:, 1])
    r_m = np.mean(np.sqrt((points[:, 0] - x_m)**2 + (points[:, 1] - y_m)**2))
    params_init = [x_m, y_m, r_m]

    result = least_squares(distance_to_circle, params_init, args=(points,))
    return result.x

def distance_to_ellipse(params, points):
    xc, yc, a, b, theta = params
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    x_rotated = (points[:, 0] - xc) * cos_theta + (points[:, 1] - yc) * sin_theta
    y_rotated = -(points[:, 0] - xc) * sin_theta + (points[:, 1] - yc) * cos_theta
    return ((x_rotated / a)**2 + (y_rotated / b)**2 - 1)

def fit_ellipse(points):
    x_m = np.mean(points[:, 0])
    y_m = np.mean(points[:, 1])
    a_m = np.std(points[:, 0]) * 2
    b_m = np.std(points[:, 1]) * 2
    theta_m = 0
    params_init = [x_m, y_m, a_m, b_m, theta_m]

    result = least_squares(distance_to_ellipse, params_init, args=(points,))
    return result.x

def detect_shapes(path_XYs, min_points=10, max_distance=5, min_samples=5, circle_threshold=0.1, ellipse_threshold=0.15):
    all_points = np.vstack([XY for XYs in path_XYs for XY in XYs])

    clustering = DBSCAN(eps=max_distance, min_samples=min_samples).fit(all_points)
    labels = clustering.labels_

    circles = []
    ellipses = []

    for label in np.unique(labels):
        if label == -1:
            continue

        cluster_points = all_points[labels == label]

        if len(cluster_points) < min_points:
            continue

        # Try fitting a circle
        circle_params = fit_circle(cluster_points)
        circle_errors = np.abs(distance_to_circle(circle_params, cluster_points))
        circle_error_ratio = np.mean(circle_errors) / circle_params[2]

        # Try fitting an ellipse
        ellipse_params = fit_ellipse(cluster_points)
        ellipse_errors = np.abs(distance_to_ellipse(ellipse_params, cluster_points))
        ellipse_error_ratio = np.mean(ellipse_errors) / np.mean(ellipse_params[2:4])

        if circle_error_ratio < circle_threshold and circle_error_ratio <= ellipse_error_ratio:
            if is_path_matching(circle_params, cluster_points, circle=True):
                circles.append(circle_params)
        elif ellipse_error_ratio < ellipse_threshold:
            if is_path_matching(ellipse_params, cluster_points, circle=False):
                ellipses.append(ellipse_params)

    return circles, ellipses

def is_path_matching(params, cluster_points, circle=True):
    if circle:
        xc, yc, r = params
        distances = np.sqrt((cluster_points[:, 0] - xc)**2 + (cluster_points[:, 1] - yc)**2)
        inliers = np.abs(distances - r) < 1.0  # Adjust the threshold as needed
    else:
        xc, yc, a, b, theta = params
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        x_rotated = (cluster_points[:, 0] - xc) * cos_theta + (cluster_points[:, 1] - yc) * sin_theta
        y_rotated = -(cluster_points[:, 0] - xc) * sin_theta + (cluster_points[:, 1] - yc) * cos_theta
        distances = (x_rotated / a)**2 + (y_rotated / b)**2
        inliers = np.abs(distances - 1) < 0.1  # Adjust the threshold as needed

    # Check if a sufficient percentage of points match the path
    match_ratio = np.sum(inliers) / len(cluster_points)
    return match_ratio > 0.1  # Adjust the threshold as needed

# Main execution
csv_path = 'problems/frag2.csv'
path_XYs = read_csv(csv_path)

# Detect shapes
circles, ellipses = detect_shapes(path_XYs)

# Plot results
plot(path_XYs, circles, ellipses)

print(f"Detected {len(circles)} circles and {len(ellipses)} ellipses.")
