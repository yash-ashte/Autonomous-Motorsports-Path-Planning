import cv2
import numpy as np
import glob
import os
import re


def extract_num(path):
    m = re.search(r"(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else 0


def get_road_mask(img):
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 80])
    upper = np.array([180, 55, 220])
    mask = cv2.inRange(hsv, lower, upper)
    roi = np.array([[
        (int(w * 0.05), h),
        (int(w * 0.95), h),
        (int(w * 0.75), int(h * 0.45)),
        (int(w * 0.25), int(h * 0.45)),
    ]], np.int32)
    roi_mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(roi_mask, roi, 255)
    return cv2.bitwise_and(mask, roi_mask)


def get_edges(img):
    mask = get_road_mask(img)
    road = cv2.bitwise_and(img, img, mask=mask)
    gray = cv2.cvtColor(road, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 110)
    h = edges.shape[0]
    edges[:int(h * 0.45)] = 0
    edges[int(h * 0.95):] = 0
    return edges, mask


def compute_center_offset(edges):
    h, w = edges.shape
    ys = np.arange(int(h * 0.55), h, 6)
    centers = []
    for y in ys:
        xs = np.where(edges[y] > 0)[0]
        if len(xs) < 2:
            continue
        centers.append((xs[0] + xs[-1]) // 2)
    if not centers:
        return None
    return (np.mean(centers) - w / 2) / (w / 2)


def close_track_loop(points, thresh=40):
    if len(points) < 10:
        return points
    p0 = points[0]
    p_end = points[-1]
    if np.linalg.norm(p0 - p_end) < thresh:
        return points
    return np.vstack([points, p0])


def create_track_video(folder, output="track_map.avi"):
    files = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        files.extend(glob.glob(os.path.join(folder, ext)))
    files = sorted(files, key=extract_num)
    if not files:
        print("No images found.")
        return

    x, y = 0.0, 0.0
    heading = 0.0
    pts = []

    for f in files:
        img = cv2.imread(f)
        edges, _ = get_edges(img)
        offset = compute_center_offset(edges)
        if offset is None:
            continue
        steering = offset * 0.035
        heading += steering
        x += np.cos(heading)
        y += np.sin(heading)
        pts.append([x, y])

    pts = np.array(pts)
    pts = close_track_loop(pts)
    pts -= pts.min(axis=0)
    scale = 900.0 / np.ptp(pts, axis=0).max()
    pts = (pts * scale).astype(int)

    sample = cv2.imread(files[0])
    h_cam, w_cam = sample.shape[:2]
    w_map = 1000
    h_map = 600
    H = max(h_cam, h_map)
    W = w_cam + w_map
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(output, fourcc, 20, (W, H))
    idx = 0

    for i, f in enumerate(files):
        img = cv2.imread(f)
        edges, mask = get_edges(img)
        mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        green = np.zeros_like(mask_3)
        green[:, :, 1] = 180
        overlay = img.copy()
        overlay[mask > 0] = overlay[mask > 0] * 0.4 + green[mask > 0] * 0.6
        map_img = np.zeros((h_map, w_map, 3), dtype=np.uint8)

        for j in range(1, len(pts)):
            cv2.line(map_img, tuple(pts[j - 1]), tuple(pts[j]), (80, 80, 80), 12)

        n = min(idx, len(pts))
        for j in range(1, n):
            cv2.line(map_img, tuple(pts[j - 1]), tuple(pts[j]), (0, 220, 0), 10)

        if idx < len(pts):
            cx, cy = pts[idx]
        else:
            cx, cy = pts[-1]

        cv2.circle(map_img, (cx, cy), 12, (0, 255, 255), -1)
        hx = int(cx + 40 * np.cos(heading))
        hy = int(cy + 40 * np.sin(heading))
        cv2.line(map_img, (cx, cy), (hx, hy), (255, 255, 255), 4)

        frame = np.zeros((H, W, 3), dtype=np.uint8)
        frame[:h_cam, :w_cam] = overlay
        frame[:h_map, w_cam:w_cam + w_map] = map_img
        writer.write(frame)
        idx += 1
        print(f"Frame {i+1}/{len(files)}")

    writer.release()
    print(f"Saved video → {output}")


if __name__ == "__main__":
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else "images"
    create_track_video(folder)