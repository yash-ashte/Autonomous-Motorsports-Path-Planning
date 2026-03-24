import cv2
import numpy as np


def get_canny_edges(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5,5), 0)

    edges = cv2.Canny(blur, 50, 150)

    h, w = edges.shape

    edges[:int(h*0.5), :] = 0
    edges[int(h*0.7):, :] = 0


    return edges

def bird_eye_transform(img, edges):

    h, w = img.shape[:2]

    ys, xs = np.where(edges > 0)

    left_mask = xs < w // 2
    right_mask = xs >= w // 2

    left_x = xs[left_mask]
    left_y = ys[left_mask]
    right_x = xs[right_mask]
    right_y = ys[right_mask]

    left_top = (left_x[np.argmin(left_y)], left_y[np.argmin(left_y)])
    left_bottom = (left_x[np.argmax(left_y)], left_y[np.argmax(left_y)])

    right_top = (right_x[np.argmin(right_y)], right_y[np.argmin(right_y)])
    right_bottom = (right_x[np.argmax(right_y)], right_y[np.argmax(right_y)])

    src = np.float32([
        left_bottom,
        right_bottom,
        right_top,
        left_top
    ])

    dst = np.float32([
        [200, h],
        [w-200, h],
        [w-200, 0],
        [200, 0]
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    bird = cv2.warpPerspective(img, M, (w, h))

    vis = img.copy()
    for p in [left_top, left_bottom, right_top, right_bottom]:
        cv2.circle(vis, p, 8, (0,0,255), -1)

    return bird, vis


def calcRaceline(birdseye):
    edges = get_canny_edges(birdseye)
    h,w = edges.shape
    ys, xs = np.where(edges > 0)
    left_mask = xs < w // 2
    right_mask = xs >= w // 2
    left_x = xs[left_mask]
    left_y = ys[left_mask]
    right_x = xs[right_mask]
    right_y = ys[right_mask]
    left_edge = {}
    right_edge = {}

    for x, y in zip(left_x, left_y):
        if y not in left_edge or x > left_edge[y]:
            left_edge[y] = x

    for x, y in zip(right_x, right_y):
        if y not in right_edge or x < right_edge[y]:
            right_edge[y] = x

    raceline = []

    for y in range(h):
        if y in left_edge and y in right_edge:
            x_center = int((left_edge[y] + right_edge[y]) / 2)
            raceline.append((x_center, y))

    vis = birdseye.copy()

    for x, y in raceline:
        cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)

    return vis, edges

img = cv2.imread("road.jpg")

edges = get_canny_edges(img)

bird, points_vis = bird_eye_transform(img, edges)

raceline_img, bird_edges = calcRaceline(bird)

cv2.imshow("Edges", edges)
cv2.imshow("Selected Points", points_vis)
cv2.imshow("Bird Eye View", bird)
cv2.imshow("Raceline", raceline_img)

cv2.waitKey(0)
cv2.destroyAllWindows()