import cv2
import numpy as np
from numpy.core.shape_base import block

def corner_harris(img, block_size, ksize=3, k=0.05):
    img_gray = np.float32(img)/255.
    ix = cv2.Sobel(img_gray, -1, 1, 0, ksize=ksize)
    iy = cv2.Sobel(img_gray, -1, 0, 1, ksize=ksize)
    ixx = cv2.filter2D(ix*ix, -1, np.ones((block_size, block_size)))/block_size/block_size
    iyy = cv2.filter2D(iy*iy, -1, np.ones((block_size, block_size)))/block_size/block_size
    ixy = iyx = cv2.filter2D(ix*iy, -1, np.ones((block_size, block_size)))/block_size/block_size
    det = ixx*iyy - (ixy)**2
    trace = ixx+iyy
    dst = det - trace*k
    return dst


cv2.samples.addSamplesDataSearchPath("../test_images")
img = cv2.imread(cv2.samples.findFile("blox.jpg"))
# img = cv2.imread(cv2.samples.findFile("chessboard.png"))
# img = cv2.resize(img, (256, 256))
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)




dst = corner_harris(img_gray, block_size=2, ksize=3, k=0.04)
img2 = img.copy()
img[dst > 0.01*dst.max()] = [0, 0, 255]
print(dst.max(), dst.min())
dst2 = cv2.cornerHarris(img_gray, 2, 3, 0.04)
print(dst2.max(), dst2.min())
img2[dst2 > 0.01*dst2.max()] = [0, 0, 255]

print("Difference between two methods = ", np.sum(cv2.absdiff(img, img2)))

cv2.imshow("Harris from scratch", img)
cv2.imshow("Harris using opencv", img2)
cv2.waitKey()
