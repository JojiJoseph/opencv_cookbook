import cv2
import numpy as np

cv2.samples.addSamplesDataSearchPath("../test_images")

fruits = cv2.imread(cv2.samples.findFile('fruits.jpg'))
fruits_lab = cv2.cvtColor(fruits, cv2.COLOR_BGR2LAB)

cv2.namedWindow("Image")

def do_nothing(value):
    pass

cv2.createTrackbar("low_l", "Image", 0, 255, do_nothing)
cv2.createTrackbar("low_a", "Image", 0, 255, do_nothing)
cv2.createTrackbar("low_b", "Image", 0, 255, do_nothing)

cv2.createTrackbar("high_l", "Image", 255, 255, do_nothing)
cv2.createTrackbar("high_a", "Image", 255, 255, do_nothing)
cv2.createTrackbar("high_b", "Image", 255, 255, do_nothing)

while True:
    low_l = cv2.getTrackbarPos("low_l", "Image")
    low_a = cv2.getTrackbarPos("low_a", "Image")
    low_b = cv2.getTrackbarPos("low_b", "Image")

    high_l = cv2.getTrackbarPos("high_l", "Image")
    high_a = cv2.getTrackbarPos("high_a", "Image")
    high_b = cv2.getTrackbarPos("high_b", "Image")

    mask = cv2.inRange(fruits_lab, np.array([low_l, low_a, low_b]), np.array([high_l, high_a, high_b]))

    segmented_fruits = cv2.bitwise_and(fruits, fruits, mask=mask)
    cv2.imshow("Image", segmented_fruits)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
