"""
A demo of moving a gizmo interactively
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

RED = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)

STATE_IDLE = 0
STATE_ON_DRAG = 1

x_offset = 0
y_offset = 0

tool_state = STATE_IDLE

gizmo = {
    "center": [400, 400],
    "arrow_length": 50,
    "is_x_active": False,
    "is_y_active": False
}

cv2.namedWindow("Move Gizmo")

def line_seg_length(start_x, start_y, end_x, end_y):
    return ((start_x - end_x)**2 + (start_y-end_y)**2)**0.5

def mouse_callback(event, x, y, flags, param):
    global gizmo, tool_state, x_offset, y_offset
    
    if tool_state == STATE_IDLE:
        gizmo["is_x_active"] = gizmo["is_y_active"] = False
    

        # Check if any tool is active
        y_delta = 1000000
        x_delta = 1000000
        # X axis
        if line_seg_length(gizmo["center"][0], gizmo["center"][1], x, y) <= gizmo["arrow_length"] and line_seg_length(gizmo["center"][0]+gizmo["arrow_length"], gizmo["center"][1], x, y) <= gizmo["arrow_length"]:
            y_delta = abs(y-gizmo["center"][1]) # Assumes axis aligned gizmo
        # Y axis
        if line_seg_length(gizmo["center"][0], gizmo["center"][1], x, y) <= gizmo["arrow_length"] and line_seg_length(gizmo["center"][0], gizmo["center"][1] + gizmo["arrow_length"], x, y) <= gizmo["arrow_length"]:
            x_delta = abs(x-gizmo["center"][0])
        
        if x_delta < 5: #px
            gizmo["is_y_active"] = True
        if y_delta < 5:
            gizmo["is_x_active"] = True

        if event == cv2.EVENT_LBUTTONDOWN:
            if gizmo["is_x_active"] or gizmo["is_y_active"]:
                x_offset = x - gizmo["center"][0]
                y_offset = y - gizmo["center"][1]
                tool_state = STATE_ON_DRAG
    if event == cv2.EVENT_MOUSEMOVE:
        if tool_state == STATE_ON_DRAG:
            if gizmo["is_x_active"]:
                gizmo["center"][0] = x - x_offset
            if gizmo["is_y_active"]:
                gizmo["center"][1] = y - y_offset
    if event == cv2.EVENT_LBUTTONUP:
        tool_state = STATE_IDLE
    

cv2.setMouseCallback("Move Gizmo", mouse_callback)

def draw_gizmo(frame, gizmo):
    cv2.arrowedLine(frame, np.int0(gizmo["center"]), np.int0(gizmo["center"])+[gizmo["arrow_length"], 0], color=RED)
    cv2.arrowedLine(frame, np.int0(gizmo["center"]), np.int0(gizmo["center"])+[0, gizmo["arrow_length"]], color=GREEN)
    if gizmo["is_x_active"]:
        cv2.arrowedLine(frame, np.int0(gizmo["center"]), np.int0(gizmo["center"])+[gizmo["arrow_length"], 0], color=YELLOW, thickness=2)
    if gizmo["is_y_active"]:
        cv2.arrowedLine(frame, np.int0(gizmo["center"]), np.int0(gizmo["center"])+[0, gizmo["arrow_length"]], color=YELLOW, thickness=2)

if __name__ == "__main__":
    while True:
        frame = np.zeros((800,800,3), dtype=np.uint8)
        draw_gizmo(frame, gizmo)
        cv2.imshow("Move Gizmo", frame)
        
        key = cv2.waitKey(1000//30) & 0xFF
        if key in [27, ord('q')]:
            break
    