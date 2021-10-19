import cv2
import numpy as np

cv2.samples.addSamplesDataSearchPath("../test_images")

img = cv2.imread(cv2.samples.findFile("baboon.jpg"))

Z = img.reshape((-1,3)).astype(np.float32)


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

K = 16

compactness,labels,centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

centers = np.int8(centers)
img_out = centers[labels.flatten()].reshape(img.shape).astype(np.uint8)

cv2.imshow("Original", img)
cv2.imshow("Output", img_out)
cv2.waitKey()