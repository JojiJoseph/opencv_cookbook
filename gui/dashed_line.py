"""
Implement dashed line
"""

import cv2
from matplotlib.pyplot import draw
import numpy as np
import math

cv2.namedWindow("Dashed Line")

canvas = np.zeros((600, 800))

start_x, start_y = 0, 0
current_x, current_y = 0, 0
on_drawing = False

def dashed_line(img, pt1, pt2, color, thickness, dash_width):
    length_of_line = ((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)**0.5
    n_steps = math.ceil(length_of_line/dash_width)
    inc_x = (pt2[0]-pt1[0])/n_steps
    inc_y = (pt2[1]-pt1[1])/n_steps
    x = pt1[0]
    y = pt1[1]
    for i in range(0, n_steps-1, 2):
        cv2.line(img, (int(x),int(y)), (int(x+inc_x), int(y+inc_y)), color, thickness)
        x += 2*inc_x
        y += 2*inc_y
    if n_steps % 2 == 1:
        cv2.line(img, (int(x),int(y)), pt2, color, thickness)

def mouse_callback(event, x, y, flags, param):
    # Global variables needed to track the state of arrow tool
    global canvas, on_drawing, start_x, start_y, current_x, current_y
    
    # When user presses the mouse button
    if event == cv2.EVENT_LBUTTONDOWN:
        if not on_drawing:
            on_drawing  = True
            start_x, start_y = x, y
    
    # When user releases the mouse button
    if event == cv2.EVENT_LBUTTONUP:
        if on_drawing:
            on_drawing = False
    # No need to put it under mouse move
    current_x, current_y = x, y
    if on_drawing:
        canvas = np.zeros((600, 800))
        try:            
            dashed_line(canvas, (start_x, start_y), (current_x, current_y), (255,0,0), 1, 25)
        except ZeroDivisionError:
            pass

cv2.setMouseCallback("Dashed Line", mouse_callback)

while True:
    cv2.imshow("Dashed Line", canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
