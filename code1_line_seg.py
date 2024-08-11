import numpy as np
import matplotlib.pyplot as plt
import cv2

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

def plot_curves_and_segments(path_XYs, segments):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    colours = plt.cm.rainbow(np.linspace(0, 1, len(path_XYs)))

    # Plot original curves
    for i, XYs in enumerate(path_XYs):
        c = colours[i]
        for XY in XYs:
            if len(XY) > 0:
                ax1.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
    ax1.set_aspect('equal')
    ax1.set_title('Original Curves')

    # Plot detected line segments
    for i, XYs in enumerate(path_XYs):
        c = colours[i]
        for XY in XYs:
            if len(XY) > 0:
                ax2.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2, alpha=0.3)
    
    if segments is not None:
        for line in segments:
            x1, y1, x2, y2 = line[0]
            ax2.plot([x1, x2], [y1, y2], color='red', linewidth=2)
    
    ax2.set_aspect('equal')
    ax2.set_title('Detected Line Segments')

    plt.tight_layout()
    plt.show()

def detect_line_segments(path_XYs, rho, theta, threshold, min_line_length, max_line_gap):
    # Find the maximum x and y coordinates
    max_x = max(max(xy[:, 0].max() for XYs in path_XYs for xy in XYs) if XYs else 0 for XYs in path_XYs)
    max_y = max(max(xy[:, 1].max() for XYs in path_XYs for xy in XYs) if XYs else 0 for XYs in path_XYs)

    # Create a blank image
    img = np.zeros((int(max_y) + 1, int(max_x) + 1), dtype=np.uint8)

    # Draw curves on the image
    for XYs in path_XYs:
        for XY in XYs:
            if len(XY) > 0:  # Check if XY is not empty
                pts = XY.astype(np.int32)
                cv2.polylines(img, [pts], False, 255, 1)

    # Apply Probabilistic Hough Transform
    lines = cv2.HoughLinesP(img, rho, theta, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)

    return lines

def main(csv_path):
    path_XYs = read_csv(csv_path)

    # Initial parameters
    rho = 1
    theta = np.pi / 180
    threshold = 55
    min_line_length = 10
    max_line_gap = 100

    # while True:
    segments = detect_line_segments(path_XYs, rho, theta, threshold, min_line_length, max_line_gap)
    plot_curves_and_segments(path_XYs, segments)

    print(f"Current parameters:")
    print(f"rho: {rho}, theta: {theta}, threshold: {threshold}")
    print(f"min_line_length: {min_line_length}, max_line_gap: {max_line_gap}")
    
    if segments is None:
        print("No line segments detected. Try adjusting the parameters.")
    else:
        print(f"Number of line segments detected: {len(segments)}")
        
        # user_input = input("Enter new parameters (rho,theta,threshold,min_line_length,max_line_gap) or 'q' to quit: ")
        
        # if user_input.lower() == 'q':
        #     break
        
        # try:
        #     rho, theta, threshold, min_line_length, max_line_gap = map(float, user_input.split(','))
        # except ValueError:
        #     print("Invalid input. Please try again.")

if __name__ == "__main__":
    csv_path = "Codes/problems/occlusion2.csv"  # Replace with your CSV file path
    main(csv_path)