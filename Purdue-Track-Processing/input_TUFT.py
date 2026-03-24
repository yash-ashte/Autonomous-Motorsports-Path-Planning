import numpy as np
import pandas as pd
from pyproj import Proj
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

def generate_smooth_tumftm_input(inner_csv, outer_csv, output_name='track_input_spline.csv', num_nodes=100):
    # 1. Load and Project to UTM (Meters)
    utm_proj = Proj(proj='utm', zone=16, ellps='WGS84') 
    
    inner_df = pd.read_csv(inner_csv)
    outer_df = pd.read_csv(outer_csv)
    
    # Use first inner point as local (0,0)
    origin_lon, origin_lat = inner_df['lon'].iloc[0], inner_df['lat'].iloc[0]
    ox, oy = utm_proj(origin_lon, origin_lat)

    ix, iy = utm_proj(inner_df['lon'].values, inner_df['lat'].values)
    ox_pts, oy_pts = utm_proj(outer_df['lon'].values, outer_df['lat'].values)
    
    inner_m = np.array([ix - ox, iy - oy])
    outer_m = np.array([ox_pts - ox, oy_pts - oy])
    print(inner_m.shape, outer_m.shape)
    inner_pts_m = np.vstack((inner_m[0], inner_m[1])).T
    outer_pts_m = np.vstack((outer_m[0], outer_m[1])).T
    
    tree = KDTree(outer_pts_m)
    _, idx = tree.query(inner_pts_m)
    matched_outer = outer_pts_m[idx]
    outer_actual = np.array([matched_outer[:, 0], matched_outer[:, 1]])
    # Midpoints (Raw Centerline)
    center_x_m = (inner_pts_m[:, 0] + matched_outer[:, 0]) / 2
    center_y_m = (inner_pts_m[:, 1] + matched_outer[:, 1]) / 2
    w_right_m = np.sqrt((inner_m[0] - center_x_m)**2 + (inner_m[1] - center_y_m)**2)
    w_left_m = np.sqrt((outer_actual[0] - center_x_m)**2 + (outer_actual[1] - center_y_m)**2)
    pd.DataFrame({
        'x_m': center_x_m,
        'y_m': center_y_m,
        'w_tr_right_m': w_right_m,
        'w_tr_left_m': w_left_m
    }).to_csv("track_input_no_spline.csv", index=False)
    plt.figure(figsize=(10, 10))
    plt.plot(inner_m[0], inner_m[1], 'r-o', label='Inner Boundary', markersize=2)
    plt.plot(outer_actual[0], outer_actual[1], 'b-o', label='Outer Boundary', markersize=2)
    plt.plot(center_x_m, center_y_m, 'k-o', label='Centerline')
    plt.title('Projected Track Coordinates (Meters)')
    plt.xlabel('Local X (East) [m]')
    plt.ylabel('Local Y (North) [m]')
    plt.legend()
    plt.axis('equal') # CRITICAL: Keeps the track from looking stretched
    plt.grid(True)
    plt.savefig('track_no_spline_center.png')
    print("Plot saved as 'track_no_spline_center.png'")


    # 2. Spline each boundary independently
    # s=0 to follow points exactly, per=True for closed loops
    tck_i, _ = splprep([inner_m[0], inner_m[1]], s=0.5, per=True)
    tck_o, _ = splprep([outer_m[0], outer_m[1]], s=0.5, per=True)
    
    # Resample both to the SAME number of nodes
    u_new = np.linspace(0, 1, num_nodes)
    inner_smooth = np.array(splev(u_new, tck_i))
    outer_smooth = np.array(splev(u_new, tck_o))

    # 3. Calculate Centerline and Widths from Smooth Data
    # Since both are sampled at the same 'u' intervals, we can average directly
    inner_pts = np.vstack((inner_smooth[0], inner_smooth[1])).T
    outer_pts = np.vstack((outer_smooth[0], outer_smooth[1])).T
    
    tree = KDTree(outer_pts)
    _, idx = tree.query(inner_pts)
    matched_outer = outer_pts[idx]

    # Midpoints (Raw Centerline)
    center_x = (inner_pts[:, 0] + matched_outer[:, 0]) / 2
    center_y = (inner_pts[:, 1] + matched_outer[:, 1]) / 2
    # center_x = (inner_smooth[0] + outer_smooth[0]) / 2
    # center_y = (inner_smooth[1] + outer_smooth[1]) / 2

    plt.figure(figsize=(10, 10))
    plt.plot(inner_smooth[0], inner_smooth[1], 'r-o', label='Inner Boundary', markersize=2)
    plt.plot(outer_smooth[0], outer_smooth[1], 'b-o', label='Outer Boundary', markersize=2)
    plt.plot(center_x, center_y, 'k--', label='Centerline')
    plt.title('Projected Track Coordinates (Meters)')
    plt.xlabel('Local X (East) [m]')
    plt.ylabel('Local Y (North) [m]')
    plt.legend()
    plt.axis('equal') # CRITICAL: Keeps the track from looking stretched
    plt.grid(True)
    plt.savefig('track_spline_center.png')
    print("Plot saved as 'track_spline_center.png'")
    # Calculate widths (Euclidean distance from center to boundaries)
    w_right = np.sqrt((inner_smooth[0] - center_x)**2 + (inner_smooth[1] - center_y)**2)
    w_left = np.sqrt((outer_smooth[0] - center_x)**2 + (outer_smooth[1] - center_y)**2)

    # 4. Save to TUMFTM Format
    pd.DataFrame({
        'x_m': center_x,
        'y_m': center_y,
        'w_tr_right_m': w_right,
        'w_tr_left_m': w_left
    }).to_csv(output_name, index=False)
    
    print(f"Optimized input saved to {output_name}")

generate_smooth_tumftm_input('inner.csv', 'outer.csv')