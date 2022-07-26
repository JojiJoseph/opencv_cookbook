import time
from typing import *
import numpy as np
import cv2


def ransac_line(x: List, y: List, iterations: int = 100, eps: int = 5) -> Tuple[int, int, int, int]:
    """Fits a line using ransac.

    The line model is (vx, vy, x0, y0). In this model, (x0, y0) is a point on the line. The quantities vx and vy are the x and y components of the slop.

    Args:
        x (List): x components of the points
        y (List): y components of the points
        iterations (int, optional): Number of ransac iterations. Defaults to 100.
        eps (int, optional): The maximum distance to inliers from the line. Defaults to 5.

    Returns:
        Tuple[int, int, int, int]: The line model vx, vy, x0, y0
    """
    best_x0, best_y0, best_vx, best_vy = x[0], y[0], 1, 1  # Init model parameters
    best_n_out = len(x)  # Number of outliers
    start_time = time.time()

    # Ransac iterations
    for _ in range(iterations):
        idx, idx2 = 0, 0
        while idx == idx2:
            idx, idx2 = np.random.randint(len(x), size=(2,))
        xa, ya = x[idx], y[idx]
        xb, yb = x[idx2], y[idx2]

        # Following two lines calculate number of outliers. Vectorized for speed
        norm = np.linalg.norm([xb-xa, yb-ya])
        n_out = np.sum(
            np.abs(np.cross([xb-xa, yb-ya], np.vstack([x-xa, y-ya]).T)) > eps * norm)

        # Update model
        if n_out < best_n_out:
            best_n_out = n_out
            best_x0, best_y0 = xa, ya
            best_vx, best_vy = xb-xa, yb-ya

    end_time = time.time()
    print("time taken for ransac", end_time-start_time)
    return best_vx, best_vy, best_x0, best_y0


if __name__ == "__main__":
    window_name = "Draw some lines"
    cv2.namedWindow(window_name)

    draw_dots = True  # If true dataset is visualized as dots
    apply_on_acc = True  # If true, previous data points are not cleared
    use_backscreen = False

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
            pre_x, pre_y = x, y
            is_mouse_down = True
        elif event == cv2.EVENT_LBUTTONUP:
            is_mouse_down = False
            if x_arr and len(x_arr) > 2:
                color = colors[np.random.randint(len(colors))]
                vx, vy, x0, y0 = ransac_line(
                    np.array(x_arr), np.array(y_arr), eps=5)
                print("Line model (vx, vy, x0, y0)", vx, vy, x0, y0)
                if use_backscreen:
                    # cv2.circle(drawing_backscreen, (int(x0),int(y0)), int(r), color=(255,255,255), thickness=2)
                    cv2.line(drawing_backscreen, np.int0([x0, y0]), np.int0(
                        [x0+vx*100, y0+vy*100]), (255, 255, 255), thickness=2)
                    drawing = drawing_backscreen.copy()
                else:
                    if apply_on_acc:
                        drawing = canvas_data.copy()
                    # cv2.circle(drawing, (int(x0),int(y0)), int(r), color=color, thickness=2)
                    cv2.line(drawing, np.int0(
                        [0, y0-x0/vx*vy]), np.int0([x0+vx*(800-x0)/vx, y0+vy*(800-x0)/vx]), color, thickness=2)
                    # cv2.circle(drawing, (x0, y0), 10, (0, 0, 255), 4)
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
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            cv2.imwrite("output.png", drawing)
