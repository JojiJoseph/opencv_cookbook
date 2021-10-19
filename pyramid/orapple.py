import cv2
import numpy as np

cv2.samples.addSamplesDataSearchPath("../test_images")

img_A = cv2.imread(cv2.samples.findFile("apple.jpg"))
img_B = cv2.imread(cv2.samples.findFile("orange.jpg"))
cv2.imshow("Apple", img_A)
cv2.imshow("Orange", img_B)
rows, cols, _ = img_A.shape
direct_blend = np.concatenate([img_A[:,:cols//2,:],img_B[:,cols//2:,:]], axis=1)
cv2.imshow("Direct Blend", direct_blend)

# n level pyramid
levels = 6
pyr_A = [img_A]
img = img_A
for i in range(levels-1):
    img = cv2.pyrDown(img)
    pyr_A.append(img)

pyr_B = [img_B]
img = img_B
for i in range(levels-1):
    img = cv2.pyrDown(img)
    pyr_B.append(img)

lap_A = []
for i in range(levels-1):
    img = cv2.subtract(pyr_A[i], cv2.pyrUp(pyr_A[i+1]))
    lap_A.append(img)
lap_A.append(pyr_A[-1])

lap_B = []
for i in range(levels-1):
    img = cv2.subtract(pyr_B[i] , cv2.pyrUp(pyr_B[i+1]))
    lap_B.append(img)
lap_B.append(pyr_B[-1])

rows, cols, _ = pyr_A[-1].shape
pyr_blend = np.concatenate([pyr_A[-1][:,:cols//2,:],pyr_B[-1][:,cols//2:,:]], axis=1) 
for i in range(levels-2,-1,-1):
    rows, cols, _ = lap_A[i].shape
    pyr_blend = cv2.add(np.concatenate([lap_A[i][:,:cols//2,:],lap_B[i][:,cols//2:,:]], axis=1) , cv2.pyrUp(pyr_blend))
cv2.imshow("Pyramid Blend", pyr_blend)
cv2.waitKey()