import time
from typing import *
import numpy as np
import cv2
import math

def hough_line(x: List, y: List):
    hist = np.zeros((400,360))
    for xi, yi in zip(x, y):
        for theta in range(360):
            theta_rad = np.radians(theta)
            r = xi*math.cos(theta_rad) + yi*math.sin(theta_rad)
            if 0 <= r < 400:
                hist[int(r),theta] += 1
    a = np.argmax(hist)
    r_max = a // 360
    theta_max = a % 360
    print(r_max, theta_max)
    img = hist/hist.max()
    cv2.circle(img, (theta_max, r_max), 6, (1,1,1), thickness=1)
    # cv2.imshow("Hough",img )
    return r_max, math.radians(theta_max), img


if __name__ == "__main__":
    window_name = "Draw some lines"
    cv2.namedWindow(window_name)

    draw_dots = True  # If true dataset is visualized as dots
    apply_on_acc = False  # If true, previous data points are not cleared
    use_backscreen = False

    drawing = np.zeros((300, 400, 3), dtype=np.uint8)

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
    hist = None
    def mouse_callback(event, x, y, flags, param):
        global drawing, pre_x, pre_y, is_mouse_down, x_arr, y_arr, hist
        if event == cv2.EVENT_LBUTTONDOWN:
            pre_x, pre_y = x, y
            is_mouse_down = True
        elif event == cv2.EVENT_LBUTTONUP:
            is_mouse_down = False
            if x_arr and len(x_arr) > 2:
                color = colors[np.random.randint(len(colors))]
            
                r, theta, hist = hough_line(x_arr, y_arr)
                y0 = (1/-math.tan(theta))*0 + r/math.sin(theta)
                y400 = (1/-math.tan(theta))*400 + r/math.sin(theta)
                if use_backscreen:
                    cv2.line(drawing_backscreen, np.int0([0, y0]), np.int0(
                        [400, y400]), (255, 255, 255), thickness=2)
                    drawing = drawing_backscreen.copy()
                else:
                    if apply_on_acc:
                        drawing = canvas_data.copy()
                    cv2.line(drawing, np.int0([0, y0]), np.int0(
                        [400, y400]), color, thickness=2)
                if use_backscreen or not apply_on_acc:
                    x_arr = []
                    y_arr = []
                if apply_on_acc:
                    temp = cv2.cvtColor(canvas_data, cv2.COLOR_BGR2GRAY)
                    # cv2.imshow("temp", temp)
                    _, temp = cv2.threshold(temp, 128, 255, cv2.THRESH_BINARY)
                    xy = cv2.findNonZero(temp).squeeze()
                    # print("xy shape", xy.shape)
                    x_arr = xy[:, 0].tolist()
                    y_arr = xy[:, 1].tolist()

        elif event == cv2.EVENT_MOUSEMOVE:
            if is_mouse_down:
                if draw_dots:
                    cv2.circle(drawing, (x, y), 1, (255, 255, 255), -1)
                    cv2.circle(canvas_data, (x, y), 1, (255, 255, 255), -1)
                else:
                    cv2.line(drawing, (pre_x, pre_y),
                             (x, y), (255, 255, 255))
                    cv2.line(canvas_data, (pre_x, pre_y),
                             (x, y), (255, 255, 255))
                x_arr.append(x)
                y_arr.append(y)
                pre_x, pre_y = x, y

    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        cv2.imshow(window_name, drawing)
        if hist is not None:
            cv2.imshow("Histogram",(500*hist).astype(np.uint8))
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            cv2.imwrite("output.png", drawing)
            cv2.imwrite("histogram.png", (500*hist).astype(np.uint8))
