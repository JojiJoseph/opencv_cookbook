import cv2
import numpy as np

cv2.samples.addSamplesDataSearchPath("../test_images")

img_org = cv2.imread(cv2.samples.findFile("pca_test1.jpg"))
img = cv2.blur(img_org, (5,5))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

contours, heirarchies = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

img_out = cv2.drawContours(img_org, contours, -1, (0,0,255), 4)

for cnt in contours:
    if cv2.contourArea(cnt) > 1e5:
        continue
    cnt = np.reshape(cnt, (-1,2)).astype(float)
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(cnt, None)
    center = np.int0(mean[0])
    n = np.linalg.norm(eigenvectors[0])
    p1 = np.int0((center[0] + 100*eigenvectors[0][0]/n, center[1] + 100*eigenvectors[0][1]/n))
    n2 = np.linalg.norm(eigenvectors[0])
    p2 = np.int0((center[0] - 100*eigenvectors[1][0]/n2, center[1] - 100*eigenvectors[1][1]/n2))
    cv2.line(img_out, center, p1, (255,0,0))
    cv2.line(img_out, center, p2, (255,0,0))
cv2.imshow("",img_out)
cv2.waitKey()
