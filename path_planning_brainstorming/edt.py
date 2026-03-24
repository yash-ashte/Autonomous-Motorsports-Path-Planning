import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

def process_track_centerline(image_path):
    # 1. Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 2. INVERSION: 
    # If the track is currently black (0) and background is white (255),
    # bitwise_not turns track to white (255) and background to black (0).
    inverted_img = cv2.bitwise_not(img)
    
    # 3. Threshold to ensure it's strictly binary
    _, binary = cv2.threshold(inverted_img, 127, 255, cv2.THRESH_BINARY)
    
    # 4. Normalize for Scipy/Skimage (convert to 0 and 1)
    binary_normalized = binary // 255
    
    # 5. Euclidean Distance Transform
    # Finds distance from the white track pixels to the nearest black boundary
    dist_map = distance_transform_edt(binary_normalized)
    
    # 6. Skeletonization
    # Reduces the white track area to a 1-pixel wide line
    skeleton = skeletonize(binary_normalized)
    
    # 7. Save results
    # Convert back to 8-bit for OpenCV
    final_output = (skeleton * 255).astype(np.uint8)
    
    cv2.imwrite('inverted_track.png', inverted_img)
    cv2.imwrite('skeleton_centerline.png', final_output)
    
    print("Inversion and centerline extraction complete.")

# Run it on your specific file
process_track_centerline('filled.png')