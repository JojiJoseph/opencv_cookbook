import cv2
import numpy as np

cv2.samples.addSamplesDataSearchPath("../test_images")

X, Y = np.meshgrid(np.arange(0,512),np.arange(0,512))

Z = np.stack([X,Y], axis=-1).astype(np.float32)

print(Z.shape)

Z = Z.reshape((-1,2))

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

K = 6

compactness,labels,centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

colors = np.array([
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 0],
    [0, 255, 255],
    [255, 0, 255]
])
print(labels.shape)

centers = np.int8(centers)
img_out = colors[labels.flatten()].reshape((512,512,3)).astype(np.uint8)

# cv2.imshow("Original", img)
cv2.imshow("Output", img_out)
cv2.waitKey()