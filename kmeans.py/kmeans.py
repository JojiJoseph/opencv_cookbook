import cv2
import numpy as np
import matplotlib.pyplot as plt

cv2.samples.addSamplesDataSearchPath("../test_images")

img = cv2.imread(cv2.samples.findFile("baboon.jpg"))

Z = img.reshape((-1,3)).astype(np.float32)

Z1 = np.random.normal(0,1, (100,2))

Z2 = np.random.normal(5,2, (100,2))


Z = np.concatenate([Z1, Z2], axis=0).astype(np.float32)

np.random.shuffle(Z)


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

K = 2

compactness,labels,centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

centers = np.int8(centers)

set1 = Z[labels.flatten() == 0]
set2 = Z[labels.flatten() == 1]

plt.scatter(set1[:,0],set1[:,1])
plt.scatter(set2[:,0],set2[:,1])

plt.show()
