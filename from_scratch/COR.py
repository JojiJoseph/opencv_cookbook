from turtle import color
import cv2
from matplotlib.pyplot import draw
import numpy as np

import time
from typing import *
import numpy as np
import cv2


if __name__ == "__main__":
    window_name = "Draw some arrows"
    cv2.namedWindow(window_name)

    drawing = np.zeros((600, 800, 3), dtype=np.uint8)

    drawing_backscreen = drawing.copy()

    pre_x, pre_y = 0, 0
    is_mouse_down = False
    x_arr = []
    y_arr = []

    colors = [
        (0, 0, 255),
        (0, 255, 0),
        (255, 0, 0),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255)
    ]

    canvas_data = drawing.copy()

    def mouse_callback(event, x, y, flags, param):
        global drawing, pre_x, pre_y, is_mouse_down, x_arr, y_arr
        if event == cv2.EVENT_LBUTTONDOWN:
            # if not is_mouse_down:
            #     drawing = np.zeros((600, 800, 3), dtype=np.uint8)
            # x_arr.append(x)
            # y_arr.append(y)
            pre_x = x
            pre_y = y
            # cv2.circle(drawing, (x, y), 1, (255, 255, 255), -1)
            is_mouse_down = True
        elif event == cv2.EVENT_LBUTTONUP:
            is_mouse_down = False
            # drawing = np.zeros((600, 800, 3), dtype=np.uint8)
            x_arr.append([pre_x, x])
            y_arr.append([pre_y, y])
            for (pre_x, x),(pre_y, y) in zip(x_arr, y_arr):
                # cv2.circle(drawing, (x, y), 1, (255, 255, 255), -1)
                # cv2.arrowedLine(drawing)
                cv2.arrowedLine(drawing_backscreen, (pre_x, pre_y), (x,y), (255, 255, 255))
            if len(x_arr) >= 2:
                color = colors[np.random.randint(len(colors))]
                X = np.array(x_arr)
                Y = np.array(y_arr)
                x_m = (X[:,0] + X[:,1])/2
                y_m = (Y[:,0] + Y[:,1])/2
                b = x_m + (Y[:,0]-Y[:,1])/(X[:,0]-X[:,1])*y_m
                A = np.vstack([np.ones((len(x_arr),)),(Y[:,0]-Y[:,1])/(X[:,0]-X[:,1])]).T
                # print("x_m.shape, b.shape, A.shape", x_m.shape, b.shape, A.shape)
                # print(A)
                x_c, y_c = np.linalg.inv(A.T @ A) @ A.T @ b
                # print(x_c, y_c)
                drawing = drawing_backscreen.copy()
                cv2.circle(drawing, np.int0((x_c,y_c)),2, color=color, thickness=-1)
                for r in range(10,801,10):
                    cv2.circle(drawing, np.int0((x_c,y_c)),r, color=color)
                # Y = np.array(y_arr)

            # if len(x_arr) > 2:
            #     X = np.vstack([np.array(x_arr),np.ones((len(x_arr),))]).T
            #     Y = np.array(y_arr)
            #     W = np.linalg.inv(X.T @ X) @ X.T @ Y
            #     y800 = W @ [800, 1]
            #     y0 = W @ [0, 1]
            #     color = colors[np.random.randint(len(colors))]
            #     cv2.line(drawing, np.int0((0, y0)), np.int0((800,y800)),color=color)
            #     x_arr = []
            #     y_arr = []

        elif event == cv2.EVENT_MOUSEMOVE:
            if is_mouse_down:
                # x_arr.append(x)
                # y_arr.append(y)
                drawing = drawing_backscreen.copy()
                cv2.line(drawing, (pre_x, pre_y), (x, y), (255, 255, 255))

    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        cv2.imshow(window_name, drawing)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            cv2.imwrite("output.png", drawing)
        if key == ord('c'):
            drawing_backscreen = np.zeros((400,800), np.uint8)
            drawing = drawing_backscreen.copy()
            x_arr = []
            y_arr = []