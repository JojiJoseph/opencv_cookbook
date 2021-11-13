import cv2
import numpy as np

cv2.samples.addSamplesDataSearchPath("../test_images")
img = cv2.imread(cv2.samples.findFile("lena.jpg"))

def do_nothing(val):
    pass

cv2.namedWindow("Pencil Sketch")
cv2.createTrackbar("sigma_s","Pencil Sketch",100,200,do_nothing)
cv2.createTrackbar("sigma_r x 100","Pencil Sketch",10,100,do_nothing)
cv2.createTrackbar("shade_factor x 1000","Pencil Sketch",50,1000,do_nothing)

show_color_image = False

while True:
    sigma_s = cv2.getTrackbarPos("sigma_s", "Pencil Sketch")
    sigma_r = cv2.getTrackbarPos("sigma_r x 100", "Pencil Sketch") / 100
    shade_factor = cv2.getTrackbarPos("shade_factor x 1000", "Pencil Sketch") / 1000
    dst_gray, dst_color = cv2.pencilSketch(img, sigma_s=sigma_s, sigma_r=sigma_r, shade_factor=shade_factor)
    
    if show_color_image:
        cv2.imshow("Pencil Sketch", dst_color)
    else:
        cv2.imshow("Pencil Sketch",dst_gray)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord('c'):
        show_color_image = not show_color_image