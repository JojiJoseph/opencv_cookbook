import cv2
import numpy as np

cv2.samples.addSamplesDataSearchPath("../test_images")

img_org = cv2.imread(cv2.samples.findFile("butterfly.jpg"))

points = []
closed = True
def mouse_event(event, x, y, flags, param):
    global clicked, seed_x, seed_y, closed
    if event == cv2.EVENT_LBUTTONDOWN:
        if closed:
            points.clear()
        closed = False
    if event == cv2.EVENT_LBUTTONUP:
        if points and (points[0][0]-x)**2 + (points[0][1]-y)**2 < 400:
            points.append((points[0][0],points[0][1]))
            closed = True
        else:
            points.append([x,y])
    elif event == cv2.EVENT_LBUTTONDBLCLK:
        seed_x, seed_y = x, y
        closed = True
        
cv2.namedWindow("Poly Selection")
cv2.setMouseCallback("Poly Selection", mouse_event)

while True:
    img_to_draw = img_org.copy()
    mask = np.zeros(img_org.shape).astype(np.uint8)
    cv2.polylines(img_to_draw, [np.array(points)], closed, color=(0,0,255), thickness=4)
    if closed and points:
        cv2.fillPoly(mask, np.array([points]).astype(np.int32),(0,0,200))
        img_to_draw = cv2.addWeighted(img_to_draw, 1, mask, 0.6, 0)
    for x,y in points:
        cv2.circle(img_to_draw, (x,y), 5, (255,0,0), thickness=-1)
    cv2.imshow("Poly Selection", img_to_draw)
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    

    