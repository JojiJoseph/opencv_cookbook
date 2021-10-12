import cv2
import numpy as np

cv2.samples.addSamplesDataSearchPath("test_images")

img = cv2.imread(cv2.samples.findFile("smarties.png"))

cv2.imshow("Original", img)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("Gray", img_gray)

img_edge = cv2.Canny(img_gray, 50, 200)

cv2.imshow("Edges", img_edge)

cv2.namedWindow("Hough Circles")

while True:
    img_out = img.copy()
    circles = cv2.HoughCircles(img_edge,cv2.HOUGH_GRADIENT, 2, 20 )

    if circles is not None:
        for circle in circles:
            circle = circle[0]
            cv2.circle(img_out, (int(circle[0]),int(circle[1])), int(circle[2]), (255,0,0), 2)

    cv2.imshow("Hough Circles", img_out)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
