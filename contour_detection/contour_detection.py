import cv2
import numpy as np

cv2.samples.addSamplesDataSearchPath("../test_images")

img_org = cv2.imread(cv2.samples.findFile("pca_test1.jpg"))
img = cv2.blur(img_org, (5,5))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

contours, heirarchies = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

img_out = cv2.drawContours(img_org, contours, -1, (0,0,255), 4)
cv2.imshow("",img_out)
cv2.waitKey()