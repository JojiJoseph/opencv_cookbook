import numpy as np
import cv2
import time

def ransac_circle(x, y, iter=1000, eps=0.5):
    best_x0, best_y0 = 0, 0
    best_n_out = np.inf
    best_r = 1
    start = time.time()
    for i in range(iter):
        idx, idx2, idx3 = np.random.randint(len(x), size=(3,))
        xa, ya = x[idx], y[idx]
        xb, yb = x[idx2], y[idx2]
        xc, yc = x[idx3], y[idx3]
        A = np.array([[2*(xa-xb), 2*(ya-yb)],
                      [2*(xa-xc), 2*(ya-yc)]])
        Y = np.array([(xa**2+ya**2-xb**2-yb**2),
                     (xa**2+ya**2-xc**2-yc**2)]).reshape((-1, 1))
        try:
            x0, y0 = np.linalg.inv(A) @ Y
        except np.linalg.LinAlgError:
            continue
        r = np.sqrt((xa-x0)**2 + (ya-y0)**2)
        n_out = np.sum((r- np.sqrt((x-x0)**2 + (y-y0)**2))**2 > eps)
        if n_out < best_n_out:
            best_n_out = n_out
            best_x0, best_y0 = x0, y0
            best_r = r
    end = time.time()
    print("Time taken: ", end - start)
    return best_x0, best_y0, best_r


cv2.namedWindow("Circle")

drawing = np.zeros((600, 800, 3))


use_backscreen = False

# cv2.putText(drawing,"Please draw some circles", (100,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
drawing_backscreen = drawing.copy()

pre_x, pre_y = 0, 0
is_mouse_down = False
x_arr = []
y_arr = []

colors = [
    (0,0,255),
    (0,255,0),
    (255,0,0),
    (255,255,0),
    (0,255,255),
    (255,0,255)
]

def mouse_callback(event, x, y, flags, param):
    global drawing, pre_x, pre_y, is_mouse_down, x_arr, y_arr
    if event == cv2.EVENT_LBUTTONDOWN:
        pre_x, pre_y = x, y
        is_mouse_down = True
    elif event == cv2.EVENT_LBUTTONUP:
        is_mouse_down = False
        if x_arr and len(x_arr) > 3:
            color = colors[np.random.randint(len(colors))]
            x0, y0, r = ransac_circle(np.array(x_arr), np.array(y_arr), eps=20)
            if use_backscreen:
                cv2.circle(drawing_backscreen, (int(x0),int(y0)), int(r), color=(255,255,255), thickness=2)
                drawing = drawing_backscreen.copy()
            else:
                cv2.circle(drawing, (int(x0),int(y0)), int(r), color=color, thickness=2)
            x_arr = []
            y_arr = []

    elif event == cv2.EVENT_MOUSEMOVE:
        if is_mouse_down:
            cv2.circle(drawing, (x, y), 1, (255, 255, 255), -1)
            # cv2.line(drawing, (pre_x,pre_y), (x,y), (255,255,255))
            x_arr.append(x)
            y_arr.append(y)
            pre_x, pre_y = x, y

cv2.setMouseCallback("Circle", mouse_callback)

while True:
    cv2.imshow("Circle", drawing)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('s'):
        cv2.imwrite("output.png", drawing)