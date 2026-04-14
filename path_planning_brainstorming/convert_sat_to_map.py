import cv2
import numpy as np

def extract_track_loop(image_path):
    # 1. Load and Pre-process
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Strong Blur to merge asphalt texture (ignore gravel/pitting)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)

    # 3. Canny Edge Detection
    # These thresholds (50, 150) are standard for road edges
    edges = cv2.Canny(blurred, 50, 150)

    # 4. Dilate the edges to close tiny gaps in the lines
    kernel = np.ones((9, 9), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)

    # 5. Find the Track Loop
    # We look for all closed shapes and keep only the BIG one
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create an empty black mask
    track_mask = np.zeros_like(gray)

    if contours:
        # Sort by area and grab the biggest one
        # This is almost always the track boundary
        main_contour = max(contours, key=cv2.contourArea)
        
        # Fill the inside of the largest contour with white
        cv2.drawContours(track_mask, [main_contour], -1, 255, thickness=cv2.FILLED)
        
        # Optional: Hollow it out to get just the 'road surface' 
        # by subtracting a slightly eroded version
        eroded = cv2.erode(track_mask, kernel, iterations=3)
        road_surface = cv2.subtract(track_mask, eroded)

    cv2.imwrite('purdue_track_fixed.png', track_mask)
    print("Mask saved: Check 'purdue_track_fixed.png'")

extract_track_loop('image.png')