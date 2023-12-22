import cv2
import numpy as np

cv2.samples.addSamplesDataSearchPath("../test_images")
orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
img1 = cv2.imread(cv2.samples.findFile("aloeL.jpg"), 0)
img2 = cv2.imread(cv2.samples.findFile("aloeR.jpg"), 0)
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)
matches = bf.match(des1,des2)
matches = sorted(matches, key = lambda x:x.distance)
for match in matches:
    print(match.imgIdx, match.queryIdx, match.trainIdx, match.distance)
print(kp1[0].pt, kp2[0].pt)
kp1_points = [kp1[match.queryIdx].pt for match in matches]
kp2_points = [kp2[match.trainIdx].pt for match in matches]
F, mask = cv2.findFundamentalMat(np.array(kp1_points), np.array(kp2_points), cv2.FM_LMEDS)
# kp1_points = np.linspace([0,0], img1.shape[:2][::-2])
lines = cv2.computeCorrespondEpilines(np.array(kp1_points).reshape(-1,1,2), 1,F)
lines2 = cv2.computeCorrespondEpilines(np.array(kp2_points).reshape(-1,1,2), 2,F)
for line, line2 in zip(lines[:10], lines2[:10]):
    a, b,c = line[0]
    x1 = 0
    y1 = int((-c - a*x1)/b)
    x2 = img2.shape[1]
    y2 = int((-c - a*x2)/b)
    cv2.line(img2, (x1,y1), (x2,y2), (0,255,0), thickness=2)
    a, b,c = line2[0]
    x1 = 0
    y1 = int((-c - a*x1)/b)
    x2 = img1.shape[1]
    y2 = int((-c - a*x2)/b)
    cv2.line(img1, (x1,y1), (x2,y2), (0,255,0), thickness=2)
# cv2.polylines(img2, lines.astype(int), False, (0,255,0))
# print(F)
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.namedWindow("out", cv2.WINDOW_NORMAL)
cv2.imshow("out", img3)
cv2.waitKey(0)