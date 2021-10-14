import cv2
import numpy as np
import matplotlib.pyplot as plt

cv2.samples.addSamplesDataSearchPath("../test_images")

img_org = cv2.imread(cv2.samples.findFile("fruits.jpg"))
img_hsv = cv2.cvtColor(img_org, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(img_hsv, np.array([34, 0, 20]), np.array([82, 255, 145]))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel=np.ones((3, 3)))

hist = cv2.calcHist([img_hsv], [0], mask, [180], [0, 180])

plt.plot(hist)
plt.show()

img_out = cv2.calcBackProject([img_hsv], [0], hist, [0, 179], 1)

cv2.imshow("Mask", mask)
cv2.imshow("Back Projection", img_out)
cv2.waitKey()
