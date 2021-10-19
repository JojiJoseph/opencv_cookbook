import cv2
import numpy as np

cv2.samples.addSamplesDataSearchPath("../test_images")

img = cv2.imread(cv2.samples.findFile("apple.jpg"))
img_up = cv2.pyrUp(img)
img_down = cv2.pyrDown(img)
cv2.imshow("Original",img)
cv2.imshow("Up",img_up)
cv2.imshow("Down",img_down)
print(img.shape, img_up.shape, img_down.shape)
cv2.waitKey()