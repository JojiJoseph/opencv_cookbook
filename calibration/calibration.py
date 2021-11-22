import cv2
import numpy as np
import glob

cv2.samples.addSamplesDataSearchPath("../test_images")

img_pts = []
obj_pts = []
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


for idx in range(1,15):
    try:
        filename = cv2.samples.findFile(f"left{idx:02}.jpg")
        img = cv2.imread(filename, 0)
        ret, corners = cv2.findChessboardCorners(img, (7,6),None)
        if ret == True:
            corners = cv2.cornerSubPix(img,corners,(11,11),(-1,-1),criteria)
            img_pts.append(corners)
            obj_pts.append(objp)
        cv2.drawChessboardCorners(img, (7,6), corners, ret)
        # cv2.imshow("", img)
        # cv2.waitKey(0)
    except cv2.error:
        pass

filename = cv2.samples.findFile(f"left12.jpg")
img = cv2.imread(filename, 0)
cv2.imshow("Distorted", img)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, img.shape[::-1],None,None)

dst = cv2.undistort(img, mtx, dist, None, None)
cv2.imshow('Undistorted',dst)
cv2.waitKey()
