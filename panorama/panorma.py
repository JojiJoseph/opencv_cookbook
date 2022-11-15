import cv2
import numpy as np

# Images are taken from https://github.com/kriyeng/kornia-stitcher
img_left = cv2.imread("./bryce_left_02.png")
img_right = cv2.imread("./bryce_right_02.png")


orb_detector = cv2.ORB_create()
kp1, des1 = orb_detector.detectAndCompute(img_left, None)
kp2, des2 = orb_detector.detectAndCompute(img_right, None)

bf_matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)

matches = bf_matcher.match(des1, des2)
matches = list(matches)

matches.sort(key=lambda x:x.distance)

matches = matches[:20]

img_matches = cv2.drawMatches(img_left, kp1, img_right, kp2, matches, None)
cv2.namedWindow("Matches", cv2.WINDOW_NORMAL)
cv2.imshow("Matches", img_matches)

pts1 = [kp1[match.queryIdx].pt for match in matches]
pts2 = [kp2[match.trainIdx].pt for match in matches]

h, ret = cv2.findHomography(np.array(pts2), np.array(pts1), cv2.RANSAC, 5.0)

shape_left = np.array(img_left.shape[:2])
shape_right = img_right.shape[:2]

img_out = cv2.warpPerspective(img_right, h, (shape_left + shape_right)[::-1])

img_out[:shape_left[0],:shape_left[1]] = img_left
img_out = img_out[:int(shape_left[0]*1.25)]

cv2.namedWindow("Panorama", cv2.WINDOW_NORMAL)
cv2.imshow("Panorama", img_out)
cv2.waitKey()