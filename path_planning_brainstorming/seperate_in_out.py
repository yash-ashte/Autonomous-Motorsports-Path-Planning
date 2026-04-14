import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def separate_boundaries(nodes_file):
    # 1. Load your extracted nodes (assuming a CSV with x, y)
    # If your data is in a different format, adjust accordingly
    df = pd.read_csv(nodes_file)
    points = df[['x', 'y']].values

    # 2. Calculate the 'Centroid' of the track
    centroid = np.mean(points, axis=0)

    # 3. Calculate distance of each point from the centroid
    distances = np.linalg.norm(points - centroid, axis=1).reshape(-1, 1)

    # 4. Use K-Means (k=2) to cluster based on distance
    # One cluster will be the 'short' distances (Inner)
    # The other will be the 'long' distances (Outer)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(distances)
    labels = kmeans.labels_

    # Determine which label is inner vs outer by comparing mean distances
    if np.mean(distances[labels == 0]) < np.mean(distances[labels == 1]):
        inner_idx, outer_idx = 0, 1
    else:
        inner_idx, outer_idx = 1, 0

    inner_points = points[labels == inner_idx]
    outer_points = points[labels == outer_idx]

    # 5. Save results
    pd.DataFrame(inner_points, columns=['x', 'y']).to_csv('inner_boundary.csv', index=False)
    pd.DataFrame(outer_points, columns=['x', 'y']).to_csv('outer_boundary.csv', index=False)

    # 6. Visualization (Check your work!)
    plt.figure(figsize=(10, 8))
    plt.scatter(inner_points[:, 0], inner_points[:, 1], c='blue', label='Inner Boundary', s=10)
    plt.scatter(outer_points[:, 0], outer_points[:, 1], c='red', label='Outer Boundary', s=10)
    plt.scatter(centroid[0], centroid[1], c='black', marker='x', label='Track Centroid')
    plt.title("Track Boundary Separation")
    plt.legend()
    plt.savefig('boundary_check.png')
    print("Separation complete. Saved 'inner_boundary.csv' and 'outer_boundary.csv'.")

# If you have your nodes in an array already, pass it here
# separate_boundaries('your_nodes.csv')