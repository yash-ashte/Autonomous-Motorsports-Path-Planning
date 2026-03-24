import numpy as np
import math
import matplotlib.pyplot as plt

def run_pure_pursuit_real_world(csv_path="waypoints.csv", image_width=720, image_height=720, kart_x_offset=-40):
    
    ym_per_pix = 10.0 / 720 
    xm_per_pix = 4.0 / 400  
    
    kart_wheelbase_m = 1.04 
    Ld_meters = 3.5         

    try:
        waypoints = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    except FileNotFoundError:
        return

    kart_x = (image_width / 2) + kart_x_offset
    kart_y = image_height

    target_x = None
    target_y = None

    for i in range(len(waypoints)-1, -1, -1):
        wp_x, wp_y = waypoints[i]
        
        dist_x_m = (wp_x - kart_x) * xm_per_pix
        dist_y_m = (kart_y - wp_y) * ym_per_pix
        dist_m = math.sqrt(dist_x_m**2 + dist_y_m**2)
        
        if dist_m >= Ld_meters:
            target_x = wp_x
            target_y = wp_y
            break

    if target_x is None:
        target_x, target_y = waypoints[0]

    dx_m = (target_x - kart_x) * xm_per_pix
    dy_m = (kart_y - target_y) * ym_per_pix 
    
    alpha = math.atan2(dx_m, dy_m)
    
    delta = math.atan2(2.0 * kart_wheelbase_m * math.sin(alpha), Ld_meters)
    delta_deg = math.degrees(delta)

    plt.figure(figsize=(8, 8))
    
    plt.plot(waypoints[:, 0], waypoints[:, 1], color='cyan', linewidth=3)
    plt.plot(kart_x, kart_y, 'g^', markersize=15)
    plt.plot(target_x, target_y, 'ro', markersize=10)
    plt.plot([kart_x, target_x], [kart_y, target_y], 'r--')
    plt.plot([kart_x, kart_x], [kart_y, kart_y - (Ld_meters / ym_per_pix)], 'g--')

    circle = plt.Circle((kart_x, kart_y), Ld_meters / ym_per_pix, color='blue', fill=False, linestyle=':')
    plt.gca().add_patch(circle)

    plt.xlim(0, image_width)
    plt.ylim(image_height, 0)
    
    plt.title(f"Real-World Scaled Pure Pursuit\nTarget: ({int(target_x)}, {int(target_y)}) | Steering Angle: {delta_deg:.2f} deg")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().set_facecolor('black')
    plt.show()

    return delta, delta_deg

if __name__ == "__main__":
    run_pure_pursuit_real_world()
