import cv2
import numpy as np
import matplotlib.pyplot as plt

def test_bev_transformation():
    img = cv2.imread('track_image.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert for matplotlib
    height, width = img.shape[:2]

    # Format: [Bottom-Left, Bottom-Right, Top-Right, Top-Left]
    src_points = np.float32([
        [0, 435],   # Bottom-Left
        [655, 420],  # Bottom-Right
        [470, 360],   # Top-Right
        [185, 360]    # Top-Left
    ])

    # Where we want those trapezoid corners to move to in the final top-down image
    offset_x = 50
    dst_points = np.float32([
        [offset_x, height],          # Bottom-Left
        [width - offset_x, height],  # Bottom-Right
        [width - offset_x, 0],       # Top-Right
        [offset_x, 0]                # Top-Left
    ])

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    bev_image = cv2.warpPerspective(img, matrix, (width, height))

    # --- export image ---
    filename = "flattened_bev_output.jpg"
    cv2.imwrite(filename, bev_image)

    # --- Visualization ---
    img_with_box = img.copy()
    pts = src_points.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(img_with_box, [pts], isClosed=True, color=(255, 0, 0), thickness=4)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.imshow(img_with_box)
    ax1.set_title("Original Camera View (with ROI)")
    ax1.axis('off')

    ax2.imshow(bev_image)
    ax2.set_title("Flattened Bird's-Eye View (BEV)")
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_bev_transformation()
