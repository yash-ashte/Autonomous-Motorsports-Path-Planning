import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

def separate_boundaries_geometric(nodes_csv, seed_point):
    df = pd.read_csv(nodes_csv)
    sx, sy = seed_point[0], seed_point[1]
    
    # 1. Calculate the polar angle (theta) and distance (r) for every point
    df['angle'] = np.degrees(np.arctan2(df['y'] - sy, df['x'] - sx)) % 360
    df['dist'] = np.sqrt((df['x'] - sx)**2 + (df['y'] - sy)**2)
    
    inner_nodes = []
    outer_nodes = []
    
    # 2. Iterate through angles (0-360)
    # We group points by angle to find the inner (min dist) and outer (max dist)
    for angle in range(360):
        # Grab points in a small angular slice (e.g., 1 degree wide)
        slice_pts = df[(df['angle'] >= angle) & (df['angle'] < angle + 1)]
        
        if not slice_pts.empty:
            # The one closest to the center is Inner, furthest is Outer
            inner_nodes.append(slice_pts.loc[slice_pts['dist'].idxmin()])
            outer_nodes.append(slice_pts.loc[slice_pts['dist'].idxmax()])
            
    # 3. Save results
    pd.DataFrame(inner_nodes)[['x', 'y']].to_csv('inner_boundary.csv', index=False)
    pd.DataFrame(outer_nodes)[['x', 'y']].to_csv('outer_boundary.csv', index=False)
    
    print("Boundaries separated based on radial distance from center.")

    
    print("Boundaries separated successfully.")
    inner_df = pd.read_csv('inner_boundary_fin.csv')
    outer_df = pd.read_csv('outer_boundary_fin.csv')

    plt.figure(figsize=(10, 8))
    
    # 3. Plot Inner Boundary (e.g., in Red)
    plt.scatter(inner_df['x'], inner_df['y'], color='red', s=5, label='Inner Boundary')
    
    # 4. Plot Outer Boundary (e.g., in Blue)
    plt.scatter(outer_df['x'], outer_df['y'], color='blue', s=5, label='Outer Boundary')
    
    plt.title('Separated Track Boundaries')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    plt.show()

# Run the separation
separate_boundaries_geometric('track_edges.csv', seed_point=(134,175))