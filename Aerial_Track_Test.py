import cv2
import numpy as np
import glob
import os
import re
from collections import deque


def extract_num(path):
    m = re.search(r"(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else 0


def get_road_mask(img):
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 0, 70]), np.array([180, 45, 230]))
    mask2 = cv2.inRange(hsv, np.array([50, 80, 80]), np.array([75, 255, 220]))
    mask = cv2.bitwise_or(mask1, mask2)
    roi_mask = np.zeros((h, w), np.uint8)
    roi_mask[int(h * 0.58):int(h * 0.91), 0:int(w)] = 255
    return cv2.bitwise_and(mask, roi_mask)


def get_edges(img):
    mask = get_road_mask(img)
    road = cv2.bitwise_and(img, img, mask=mask)
    gray = cv2.cvtColor(road, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 110)
    h = edges.shape[0]
    edges[:int(h * 0.52)] = 0
    edges[int(h * 0.93):] = 0
    return edges, mask


def compute_center_offset(edges):
    h, w = edges.shape
    y_start = int(h * 0.52)
    ys = np.arange(y_start, h, 6)
    weighted_sum = 0.0
    weight_total = 0.0
    for y in ys:
        xs = np.where(edges[y] > 0)[0]
        if len(xs) < 2:
            continue
        center = (xs[0] + xs[-1]) / 2.0
        proximity = (y - y_start) / max(h - y_start, 1)
        weight = 1.0 + 4.0 * proximity
        weighted_sum += center * weight
        weight_total += weight
    if weight_total == 0:
        return None
    mean_center = weighted_sum / weight_total
    return (mean_center - w / 2) / (w / 2)


def close_track_loop(points, thresh=40):
    if len(points) < 10:
        return points
    if np.linalg.norm(points[0] - points[-1]) < thresh:
        return points
    return np.vstack([points, points[0]])


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
    steering_buf = deque(maxlen=3)

    for f in files:
        img = cv2.imread(f)
        edges, _ = get_edges(img)
        offset = compute_center_offset(edges)
        steering_buf.append(offset * 0.68 if offset is not None else 0.0)
        heading += np.mean(steering_buf)
        x += np.cos(heading)
        y += np.sin(heading)
        pts.append([x, y])

    pts = np.array(pts)
    pts = close_track_loop(pts)
    pts -= pts.min(axis=0)
    scale = 500.0 / np.ptp(pts, axis=0).max()
    pts = (pts * scale).astype(int)

    sample = cv2.imread(files[0])
    h_cam, w_cam = sample.shape[:2]
    w_map, h_map = 800, 500
    H = max(h_cam, h_map)
    W = w_cam + w_map

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(output, fourcc, 20, (W, H))

    for i, f in enumerate(files):
        img = cv2.imread(f)
        edges, mask = get_edges(img)

        mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        green = np.zeros_like(mask_3)
        green[:, :, 1] = 180
        overlay = img.copy()
        overlay[mask > 0] = overlay[mask > 0] * 0.4 + green[mask > 0] * 0.6

        offset = compute_center_offset(edges)
        if offset is not None:
            cx_cam = int((offset + 1) / 2 * w_cam)
            cv2.line(overlay, (cx_cam, h_cam - 10), (w_cam // 2, h_cam - 10), (0, 255, 255), 3)
            cv2.circle(overlay, (cx_cam, h_cam - 10), 7, (0, 100, 255), -1)

        map_img = np.zeros((h_map, w_map, 3), dtype=np.uint8)
        for j in range(1, len(pts)):
            cv2.line(map_img, tuple(pts[j - 1]), tuple(pts[j]), (80, 80, 80), 12)
        for j in range(1, min(i + 1, len(pts))):
            cv2.line(map_img, tuple(pts[j - 1]), tuple(pts[j]), (0, 220, 0), 10)

        idx = min(i, len(pts) - 1)
        cx, cy = pts[idx]
        cv2.circle(map_img, (cx, cy), 12, (0, 255, 255), -1)
        hx = int(cx + 40 * np.cos(heading))
        hy = int(cy + 40 * np.sin(heading))
        cv2.line(map_img, (cx, cy), (hx, hy), (255, 255, 255), 4)

        frame = np.zeros((H, W, 3), dtype=np.uint8)
        frame[:h_cam, :w_cam] = overlay
        frame[:h_map, w_cam:w_cam + w_map] = map_img
        writer.write(frame)
        print(f"Frame {i+1}/{len(files)}")

    writer.release()
    print(f"Saved video → {output}")


if __name__ == "__main__":
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else "images"
    create_track_video(folder)