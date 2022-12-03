import cv2
import numpy as np
import matplotlib.pyplot as plt

cv2.samples.addSamplesDataSearchPath("../test_images")
img = cv2.imread(cv2.samples.findFile("pic3.png"), 0)
_, img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("Image after thresholding", img)
cv2.waitKey()

img = cv2.medianBlur(img, 3)

skelton = np.zeros(img.shape, np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
# done = False
iteration = 0
while True:
    img_eroded = cv2.erode(img, kernel)
    # img_dilated = cv2.dilate(img_eroded, kernel)
    # delta = cv2.subtract(img, img_dilated)
    img_open = cv2.morphologyEx(img_eroded, cv2.MORPH_OPEN, kernel)
    delta = cv2.subtract(img_eroded, img_open)
    skelton = cv2.bitwise_or(skelton, delta)
    img = img_eroded
    iteration += 1
    cv2.imshow("Skelton", skelton)
    cv2.waitKey(200)
    if cv2.countNonZero(img) == 0:
        break
print("Total number of iterations = ", iteration)
cv2.waitKey()
