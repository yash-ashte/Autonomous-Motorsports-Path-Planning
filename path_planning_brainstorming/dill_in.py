import cv2
import numpy as np

def fill_track_rows(image_path, output_path='filled_track.png'):
    # 1. Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 2. Ensure image is binary (0 for black, 255 for white)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # 3. Create a copy to modify
    filled = binary.copy()
    
    # 4. Iterate row by row
    for y in range(binary.shape[0]):
        # Get indices of all black pixels (0) in the current row
        black_pixels = np.where(binary[y] == 0)[0]
        
        for i in range(len(black_pixels) - 2):
            # Find the first and last black pixel in this row
            start = black_pixels[i]
            end = black_pixels[i + 1]
            filled[y, start:end] = 0
            i += 2
            
            # Set all pixels between the first and last black pixel to black
            
            
    cv2.imwrite(output_path, filled)
    print(f"Filled image saved as '{output_path}'")

# Run the function
fill_track_rows('track_edge.png')