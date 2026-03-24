import cv2
import numpy as np
import os
import glob


def get_road_mask(img):
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_asphalt = np.array([0,   0,  80])
    upper_asphalt = np.array([180, 55, 220])
    color_mask = cv2.inRange(hsv, lower_asphalt, upper_asphalt)

    roi_pts = np.array([[
        (int(w * 0.05), h),
        (int(w * 0.95), h),
        (int(w * 0.75), int(h * 0.45)),
        (int(w * 0.25), int(h * 0.45)),
    ]], dtype=np.int32)
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(roi_mask, roi_pts, 255)

    combined = cv2.bitwise_and(color_mask, roi_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  kernel)
    return combined


def get_canny_edges(img):
    road_mask = get_road_mask(img)
    road_only = cv2.bitwise_and(img, img, mask=road_mask)
    gray  = cv2.cvtColor(road_only, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 100)
    mask_edges = cv2.Canny(road_mask, 50, 150)
    combined_edges = cv2.bitwise_or(edges, mask_edges)
    h, w = combined_edges.shape
    combined_edges[:int(h * 0.45), :] = 0
    combined_edges[int(h * 0.95):, :] = 0
    return combined_edges, road_mask


def create_canny_video(
    image_folder,
    output_path="canny_track.avi",
    fps=10,
    panel_size=(426, 480),
    hold_frames=2,
):
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(image_folder, ext)))
        image_paths.extend(glob.glob(os.path.join(image_folder, ext.upper())))
    import re
    def extract_num(p):
        m = re.search(r'img-(\d+)', os.path.basename(p), re.IGNORECASE)
        return int(m.group(1)) if m else 0

    image_paths = sorted(set(image_paths), key=extract_num)

    if not image_paths:
        print(f"No images found in '{image_folder}'")
        return

    pw, ph = panel_size
    vw = pw * 3
    vh = ph

    if not output_path.lower().endswith(".avi"):
        output_path = output_path.rsplit(".", 1)[0] + ".avi"

    print(f"Found {len(image_paths)} images.")
    print(f"Output: {output_path}  |  Frame size: {vw}x{vh}  |  FPS: {fps}")

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (vw, vh))

    if not writer.isOpened():
        print("ERROR: VideoWriter failed to open.")
        print("  Try: pip install opencv-python --upgrade")
        return

    print("VideoWriter opened OK. Processing frames…")
    frames_written = 0

    for i, path in enumerate(image_paths):
        filename = os.path.basename(path)
        img = cv2.imread(path)
        if img is None:
            print(f"  [{i+1:3d}] SKIP (unreadable) {filename}")
            continue

        edges, road_mask = get_canny_edges(img)
        h, w = img.shape[:2]

        orig_vis = img.copy()
        roi_pts = np.array([[
            (int(w * 0.05), h),
            (int(w * 0.95), h),
            (int(w * 0.75), int(h * 0.45)),
            (int(w * 0.25), int(h * 0.45)),
        ]], dtype=np.int32)
        cv2.polylines(orig_vis, roi_pts, True, (0, 255, 255), 2)

        mask_vis = np.zeros((h, w, 3), dtype=np.uint8)
        mask_vis[:, :, 1] = road_mask

        edge_vis = np.zeros((h, w, 3), dtype=np.uint8)
        edge_vis[edges > 0] = (200, 220, 0)

        # Resize each panel to exact panel_size
        p1 = cv2.resize(orig_vis, (pw, ph), interpolation=cv2.INTER_AREA)
        p2 = cv2.resize(mask_vis, (pw, ph), interpolation=cv2.INTER_AREA)
        p3 = cv2.resize(edge_vis, (pw, ph), interpolation=cv2.INTER_AREA)

        frame = np.hstack([p1, p2, p3])

        assert frame.shape[:2] == (vh, vw), \
            f"Frame size mismatch: {frame.shape[:2]} vs expected ({vh}, {vw})"

        # Dividers
        cv2.line(frame, (pw,     0), (pw,     vh), (60, 60, 60), 2)
        cv2.line(frame, (pw * 2, 0), (pw * 2, vh), (60, 60, 60), 2)

        lc = dict(fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                  fontScale=0.45, thickness=1, lineType=cv2.LINE_AA)
        cv2.putText(frame, "Original + ROI",  (8, 20), color=(0, 255, 255), **lc)
        cv2.putText(frame, "Road Mask",       (pw + 8, 20), color=(80, 220, 80), **lc)
        cv2.putText(frame, "Road-only Canny", (pw * 2 + 8, 20), color=(200, 220, 0), **lc)
        cv2.putText(frame, f"[{i+1}/{len(image_paths)}] {filename}",
                    (8, vh - 8), color=(120, 120, 120), **lc)

        for _ in range(max(1, hold_frames)):
            writer.write(frame)
            frames_written += 1

        print(f"  [{i+1:3d}/{len(image_paths)}] {filename}")

    writer.release()
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nDone! {frames_written} frames written.")
    print(f"Video saved → {os.path.abspath(output_path)}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else "images"
    output = sys.argv[2] if len(sys.argv) > 2 else "canny_track.avi"
    fps    = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    hold   = int(sys.argv[4]) if len(sys.argv) > 4 else 2

    create_canny_video(folder, output_path=output, fps=fps, hold_frames=hold)