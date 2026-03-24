import cv2
import numpy as np


img = cv2.imread("road.jpg")

h, w = img.shape[:2]


src = np.float32([
    [60, 420],#bl
    [620, 420],#br
    [350, 320],#tr
    [280, 320]#tl
])

dst = np.float32([
    [200, 480],
    [440, 480],
    [440, 0],
    [200, 0]
])


M = cv2.getPerspectiveTransform(src, dst)


bird_eye = cv2.warpPerspective(img, M, (w, h))


for p in src:
    cv2.circle(img, tuple(p.astype(int)), 8, (0,0,255), -1)


cv2.imshow("Original", img)
cv2.imshow("Bird Eye View", bird_eye)

cv2.waitKey(0)
cv2.destroyAllWindows()