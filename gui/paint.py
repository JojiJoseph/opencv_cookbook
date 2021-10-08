"""
Implements drawing using mouse pointer
"""

import cv2
import numpy as np

cv2.namedWindow("Paint")

drawing = np.zeros((600, 800))

pre_x, pre_y = 0, 0
is_mouse_down = False

def mouse_callback(event, x, y, flags, param):
    global drawing, pre_x, pre_y, is_mouse_down
    if event == cv2.EVENT_LBUTTONDOWN:
        pre_x, pre_y = x, y
        is_mouse_down = True
    elif event == cv2.EVENT_LBUTTONUP:
        is_mouse_down = False
    elif event == cv2.EVENT_MOUSEMOVE:
        if is_mouse_down:
            cv2.line(drawing, (pre_x,pre_y), (x,y), (255,0,0))
            pre_x, pre_y = x, y
    elif event == cv2.EVENT_MOUSEWHEEL:
        print(x,y)

cv2.setMouseCallback("Paint", mouse_callback)

while True:
    cv2.imshow("Paint", drawing)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break