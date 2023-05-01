# Didn't know this feature is available already

import cv2

cv2.samples.addSamplesDataSearchPath("../test_images")

img_org = cv2.imread(cv2.samples.findFile("fruits.jpg"))

r = cv2.selectROI("Select a bounding box", img_org)

img_cropped = img_org[r[1]: r[1] + r[3],r[0]:r[0] + r[2]]

cv2.imshow("Cropped", img_cropped)
cv2.waitKey()