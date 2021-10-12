import cv2
import numpy as np

cv2.samples.addSamplesDataSearchPath("../test_images")

img = cv2.imread(cv2.samples.findFile("building.jpg"))

cv2.imshow("Original", img)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("Gray", img_gray)

img_edge = cv2.Canny(img_gray, 50, 200)

cv2.imshow("Edges", img_edge)

cv2.namedWindow("Hough Lines")
def do_nothing(value):
    pass
cv2.createTrackbar("Threshold", "Hough Lines", 50, 200, do_nothing)

while True:
    img_out = img.copy()
    thresh = cv2.getTrackbarPos("Threshold", "Hough Lines")
    lines = cv2.HoughLinesP(img_edge,1,np.pi/180, thresh )

    if lines is not None:
        for line in lines:
            line = line[0]
            cv2.line(img_out, (line[0],line[1]), (line[2],line[3]), (255,0,0), 2)

    cv2.imshow("Hough Lines", img_out)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
