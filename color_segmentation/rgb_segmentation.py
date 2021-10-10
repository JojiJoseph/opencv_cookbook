import cv2
import numpy as np

cv2.samples.addSamplesDataSearchPath("../test_images")

fruits = cv2.imread(cv2.samples.findFile('fruits.jpg'))
fruits_rgb = cv2.cvtColor(fruits, cv2.COLOR_BGR2RGB)

cv2.namedWindow("Image")

def do_nothing(value):
    pass

cv2.createTrackbar("low_r", "Image", 0, 255, do_nothing)
cv2.createTrackbar("low_g", "Image", 0, 255, do_nothing)
cv2.createTrackbar("low_b", "Image", 0, 255, do_nothing)

cv2.createTrackbar("high_r", "Image", 255, 255, do_nothing)
cv2.createTrackbar("high_g", "Image", 255, 255, do_nothing)
cv2.createTrackbar("high_b", "Image", 255, 255, do_nothing)

while True:
    low_r = cv2.getTrackbarPos("low_r", "Image")
    low_g = cv2.getTrackbarPos("low_g", "Image")
    low_b = cv2.getTrackbarPos("low_b", "Image")

    high_r = cv2.getTrackbarPos("high_r", "Image")
    high_g = cv2.getTrackbarPos("high_g", "Image")
    high_b = cv2.getTrackbarPos("high_b", "Image")

    mask = cv2.inRange(fruits_rgb, np.array([low_r, low_g, low_b]), np.array([high_r, high_g, high_b]))

    segmented_fruits = cv2.bitwise_and(fruits, fruits, mask=mask)
    cv2.imshow("Image", segmented_fruits)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
