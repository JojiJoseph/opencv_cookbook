"""
Implements drawing arrow using mouse pointer
"""

import cv2
from matplotlib.pyplot import draw
import numpy as np
import math

cv2.namedWindow("Arrow")

canvas = np.zeros((600, 800))

start_x, start_y = 0, 0
current_x, current_y = 0, 0
on_drawing = False

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
            # Scale factor is used to limit the length of the arrow
            scale_factor = 200/math.sqrt((current_x - start_x)**2 + (start_y-current_y)**2)
            
            # Use scale factor only the arrow length is more than 200.
            # Not necessary. But will give good user experience
            if scale_factor > 1:
                scale_factor = 1
            
            cv2.arrowedLine(canvas, (start_x, start_y), np.int0((start_x + (current_x-start_x)*scale_factor, start_y + (current_y-start_y)*scale_factor)), (255,0,0), 2)
        except ZeroDivisionError:
            pass

cv2.setMouseCallback("Arrow", mouse_callback)

while True:
    cv2.imshow("Arrow", canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
