import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.optimize import minimize_scalar
from sklearn.cluster import DBSCAN
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

def preprocess_curves(path_XYs):
    return np.vstack([XY for XYs in path_XYs for XY in XYs])

def cluster_points(points, eps=5, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(points)
    return labels

def compute_centroid(points):
    return np.mean(points, axis=0)

def generate_candidate_axes(centroid, num_axes=36):
    angles = np.linspace(0, np.pi, num_axes)
    return [(centroid, np.array([np.cos(angle), np.sin(angle)])) for angle in angles]

def reflect_point(point, line_point, line_direction):
    v = line_direction / np.linalg.norm(line_direction)
    w = point - line_point
    return point - 2 * (np.dot(w, v) * v - w)

def evaluate_symmetry(points, line_point, line_direction, max_distance):
    reflected_points = np.array([reflect_point(p, line_point, line_direction) for p in points])
    distances = cdist(points, reflected_points)
    symmetry_score = np.sum(np.min(distances, axis=1) <= max_distance) / len(points)
    return symmetry_score

def find_best_symmetry_axis(points, candidate_axes, max_distance):
    best_score = 0
    best_axis = None
    for axis in candidate_axes:
        score = evaluate_symmetry(points, axis[0], axis[1], max_distance)
        if score > best_score:
            best_score = score
            best_axis = axis
    return best_axis, best_score

def optimize_max_distance(points, candidate_axes):
    def objective(max_distance):
        _, score = find_best_symmetry_axis(points, candidate_axes, max_distance)
        return -score  # Minimize negative score to maximize score
    
    result = minimize_scalar(objective, bounds=(0.1, 10), method='bounded')
    return result.x

def detect_symmetry_multiple(path_XYs, eps=5, min_samples=15):
    points = preprocess_curves(path_XYs)
    labels = cluster_points(points, eps, min_samples)
    
    symmetry_axes = []
    for label in np.unique(labels):
        if label == -1:  # Skip noise points
            continue
        cluster_points1 = points[labels == label]
        if len(cluster_points1) < min_samples:
            continue  # Skip clusters that are too small
        
        centroid = compute_centroid(cluster_points1)
        candidate_axes = generate_candidate_axes(centroid)
        
        max_distance = optimize_max_distance(cluster_points1, candidate_axes)
        best_axis, best_score = find_best_symmetry_axis(cluster_points1, candidate_axes, max_distance)
        
        symmetry_axes.append((best_axis, best_score, max_distance))
    
    return symmetry_axes

def plot_with_multiple_symmetry(path_XYs, symmetry_axes):
    fig, ax = plt.subplots(tight_layout=True, figsize=(10, 10))
    colours = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    
    for i, XYs in enumerate(path_XYs):
        c = colours[i % len(colours)]
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
    
    for i, (symmetry_axis, score, _) in enumerate(symmetry_axes):
        line_point, line_direction = symmetry_axis
        x_range = ax.get_xlim()
        y_range = ax.get_ylim()
        t = np.linspace(-100, 100, 1000)
        x = line_point[0] + t * line_direction[0]
        y = line_point[1] + t * line_direction[1]
        mask = (x >= x_range[0]) & (x <= x_range[1]) & (y >= y_range[0]) & (y <= y_range[1])
        ax.plot(x[mask], y[mask], '--', linewidth=2, label=f'Symmetry Axis {i+1} (score: {score:.2f})')
    
    ax.set_aspect('equal')
    ax.legend()
    plt.show()

def save_detected_shapes(path_XYs, labels, original_file):
    base_name = os.path.splitext(os.path.basename(original_file))[0]
    output_file = f"{base_name}_sol.csv"
    output_path = os.path.join(os.path.dirname(original_file), output_file)

    all_points = np.vstack([XY for XYs in path_XYs for XY in XYs])
    
    with open(output_path, 'w') as f:
        # Write original shapes
        for i, XYs in enumerate(path_XYs):
            for j, XY in enumerate(XYs):
                for k, point in enumerate(XY):
                    f.write(f"original,{i},{j},{k},{point[0]},{point[1]}\n")
        
        # Write detected shapes
        label_index = 0
        for label in np.unique(labels):
            if label == -1:  # Skip noise points
                continue
            shape_points = all_points[labels == label]
            if len(shape_points) > 0:
                for k, point in enumerate(shape_points):
                    f.write(f"detected,{label_index},{k},{point[0]},{point[1]}\n")
                label_index += 1

    print(f"Original and detected shapes saved to: {output_path}")

# Main execution
if __name__ == "__main__":
    csv_path = 'problems/isolated.csv'  # Replace with your actual CSV file path
    path_XYs = read_csv(csv_path)
    
    # You may need to adjust these parameters based on your data
    eps = 5
    min_samples = 5
    
    symmetry_axes = detect_symmetry_multiple(path_XYs, eps, min_samples)
    
    points = preprocess_curves(path_XYs)
    labels = cluster_points(points, eps, min_samples)
    if not symmetry_axes:
        print("No symmetry axes detected. Try adjusting the clustering parameters.")
    else:
        for i, (_, score, max_distance) in enumerate(symmetry_axes):
            print(f"Symmetry axis {i+1}:")
            print(f"  Score: {score:.2f}")
            print(f"  Max distance: {max_distance:.2f}")
        
        plot_with_multiple_symmetry(path_XYs, symmetry_axes)

    
    save_detected_shapes(path_XYs, labels, csv_path)