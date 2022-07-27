import time
import numpy as np
import cv2

draw_dots = True
apply_on_acc = True
window_name = "Please draw some lines"

def ransac_line(x, y, iter=1000, eps=0.5):
    best_x0, best_y0, best_vx, best_vy = 0, 0, 1, 1
    best_n_out = np.inf
    start = time.time()
    for i in range(iter):
        idx, idx2 = 0, 0
        while idx == idx2:
            idx, idx2 = np.random.randint(len(x), size=(2,))
        xa, ya = x[idx], y[idx]
        xb, yb = x[idx2], y[idx2]
        n_out = 0
        norm = np.linalg.norm([xb-xa,yb-ya])
        for i in range(len(x)):
            if np.abs(np.cross([xb-xa,yb-ya],[x[i]-xa,y[i]-ya])) > eps * norm:
                n_out += 1
        if n_out < best_n_out:
            best_n_out = n_out
            best_x0, best_y0 = xa, ya
            best_vx, best_vy = xb-xa, yb-ya
    end = time.time()
    print("time taken for ransac", end-start)
    return best_vx, best_vy, best_x0, best_y0


cv2.namedWindow(window_name)

drawing = np.zeros((600, 800, 3),dtype=np.uint8)


use_backscreen = False

cv2.putText(drawing, window_name, (100,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
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
            vx,vy,x0,y0 = ransac_line(np.array(x_arr), np.array(y_arr), eps=20)
            if use_backscreen:
                # cv2.circle(drawing_backscreen, (int(x0),int(y0)), int(r), color=(255,255,255), thickness=2)
                # cv2.line(drawing_backscreen, np.int0([x0,y0]), np.int0([x0+vx*100,y0+vy*100]),(255,255,255),thickness=2)
                cv2.line(drawing_backscreen, np.int0([0,y0-x0/vx*vy]), np.int0([x0+vx*(800-x0)/vx,y0+vy*(800-x0)/vx]),color,thickness=2)
                drawing = drawing_backscreen.copy()
            else:
                if apply_on_acc:
                    drawing = canvas_data.copy()
                # cv2.circle(drawing, (int(x0),int(y0)), int(r), color=color, thickness=2)
                cv2.line(drawing, np.int0([0,y0-x0/vx*vy]), np.int0([x0+vx*(800-x0)/vx,y0+vy*(800-x0)/vx]),color,thickness=2)
            if use_backscreen or not apply_on_acc:
                x_arr = []
                y_arr = []
            if apply_on_acc:
                temp = cv2.cvtColor(canvas_data,cv2.COLOR_BGR2GRAY)
                xy = cv2.findNonZero(temp).squeeze()
                print("xy shape", xy.shape)
                x_arr = xy[:,0].tolist()
                y_arr = xy[:,1].tolist()

    elif event == cv2.EVENT_MOUSEMOVE:
        if is_mouse_down:
            if draw_dots:
                cv2.circle(drawing, (x,y), 1, (255,255,255), -1)
                cv2.circle(canvas_data, (x,y), 1, (255,255,255), -1)
            else:
                cv2.line(canvas_data, (pre_x,pre_y), (x,y), (255,255,255))
            x_arr.append(x)
            y_arr.append(y)
            pre_x, pre_y = x, y

cv2.setMouseCallback(window_name, mouse_callback)

while True:
    cv2.imshow(window_name, drawing)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break