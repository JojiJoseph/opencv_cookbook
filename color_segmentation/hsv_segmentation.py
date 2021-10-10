import cv2
import numpy as np

cv2.samples.addSamplesDataSearchPath("../test_images")

fruits = cv2.imread(cv2.samples.findFile('fruits.jpg'))
fruits_hsv = cv2.cvtColor(fruits, cv2.COLOR_BGR2HSV)

cv2.namedWindow("Image")

def do_nothing(value):
    pass

cv2.createTrackbar("low_h", "Image", 0, 179, do_nothing)
cv2.createTrackbar("low_s", "Image", 0, 255, do_nothing)
cv2.createTrackbar("low_v", "Image", 0, 255, do_nothing)

cv2.createTrackbar("high_h", "Image", 179, 179, do_nothing)
cv2.createTrackbar("high_s", "Image", 255, 255, do_nothing)
cv2.createTrackbar("high_v", "Image", 255, 255, do_nothing)

while True:
    low_h = cv2.getTrackbarPos("low_h", "Image")
    low_s = cv2.getTrackbarPos("low_s", "Image")
    low_v = cv2.getTrackbarPos("low_v", "Image")

    high_h = cv2.getTrackbarPos("high_h", "Image")
    high_s = cv2.getTrackbarPos("high_s", "Image")
    high_v = cv2.getTrackbarPos("high_v", "Image")

    mask = cv2.inRange(fruits_hsv, np.array([low_h, low_s, low_v]), np.array([high_h, high_s, high_v]))

    segmented_fruits = cv2.bitwise_and(fruits, fruits, mask=mask)
    cv2.imshow("Image", segmented_fruits)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
