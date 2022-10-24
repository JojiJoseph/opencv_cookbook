"""
A demo of croping an image interactively.
No load or save.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the file
cv2.samples.addSamplesDataSearchPath("../test_images")
img_org = cv2.imread(cv2.samples.findFile("lena.jpg"))

# Define states
STATE_IDLE = 0
STATE_DRAWING = 1
STATE_FINISHED = 2

# Coordinates for crop
start_x, start_y, end_x, end_y = None, None, None, None

tool_state = STATE_IDLE # Initial state

cv2.namedWindow("Interactive Crop Demo")

def mouse_callback(event, x, y, flags, param):
    global tool_state, start_x, start_y, end_x, end_y
    if event == cv2.EVENT_LBUTTONDOWN:
        # if tool_state == STATE_IDLE:
        tool_state = STATE_DRAWING
        start_x, start_y = x, y
        end_x, end_y = x, y
    if event == cv2.EVENT_LBUTTONUP:
        if tool_state == STATE_DRAWING:
            tool_state = STATE_FINISHED
            end_x, end_y = x, y
    if event == cv2.EVENT_MOUSEMOVE:
        if tool_state == STATE_DRAWING:
            end_x, end_y = x, y

cv2.setMouseCallback("Interactive Crop Demo", mouse_callback)


while True:

    img_out = img_org.copy()
    
    if tool_state == STATE_DRAWING or tool_state == STATE_FINISHED:
        cv2.rectangle(img_out, (start_x, start_y), (end_x, end_y), (255,255,255), thickness=2)
    cv2.imshow("Interactive Crop Demo", img_out)
    
    key = cv2.waitKey(1) & 0xFF
    if key in [27, ord('q')]:
        break
    if key == 13 and tool_state == STATE_FINISHED:
        start_x, end_x = min(start_x, end_x), max(start_x, end_x)
        start_y, end_y = min(start_y, end_y), max(start_y, end_y)
        img_org = img_org[start_y:end_y+1,start_x:end_x+1,:]
        tool_state = STATE_IDLE
    if key == ord('r'): # Reset
        img_org = cv2.imread(cv2.samples.findFile("lena.jpg"))
        tool_state = STATE_IDLE
