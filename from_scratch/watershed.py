import cv2
import numpy as np
from heapq import heappush, heappop
import time

cv2.samples.addSamplesDataSearchPath("../test_images")

img_org = cv2.imread(cv2.samples.findFile("fruits.jpg"))
img_gray = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)

img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
# img_gradient morphologyEx
img_grad = cv2.morphologyEx(img_gray, cv2.MORPH_GRADIENT, np.ones((3, 3), dtype=np.uint8))
img_grad = cv2.GaussianBlur(img_grad, (5, 5), 0)
img_grad = cv2.GaussianBlur(img_grad, (5, 5), 0)
img_grad = cv2.GaussianBlur(img_grad, (5, 5), 0)
img_b, img_g, img_r = cv2.split(img_org)
img_b, img_r, img_g = cv2.GaussianBlur(img_b, (5, 5), 0), cv2.GaussianBlur(img_r, (5, 5), 0), cv2.GaussianBlur(img_g, (5, 5), 0)
img_grad_b = cv2.morphologyEx(img_b, cv2.MORPH_GRADIENT, np.ones((3, 3), dtype=np.uint8))
img_grad_r = cv2.morphologyEx(img_r, cv2.MORPH_GRADIENT, np.ones((3, 3), dtype=np.uint8))
img_grad_g = cv2.morphologyEx(img_g, cv2.MORPH_GRADIENT, np.ones((3, 3), dtype=np.uint8))
img_grad_2 = np.maximum(img_grad_b, img_grad_r, img_grad_g)
img_grad_2 = cv2.GaussianBlur(img_grad, (5, 5), 0)

img_grad = cv2.addWeighted(img_grad, 0.5, img_grad_2, 0.5, 0)

labels = np.zeros_like(img_grad)
points  = []
# mouse callback function
def mouse_callback(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONUP:
        points.append((x, y))
        print("Point added: ", x, y)
        cv2.circle(img_org, (x, y), 5, (255, 255, 255), -1)
        cv2.putText(img_org, str(len(points)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        labels[y, x] = len(points)
        # cv2.imshow("Labels", labels)
cv2.namedWindow('Original')
cv2.setMouseCallback('Original', mouse_callback)
while True:
    cv2.imshow("Original", img_org)
    cv2.imshow("Gray", img_gray)
    cv2.imshow("Gradient", img_grad)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Now we have the points, let's do the watershed
start_time = time.time()
heap = []
for x, y in points:
    heappush(heap, (img_grad[y, x], x, y))

while heap:
    _, x, y = heappop(heap)
    # 8 way connected
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), 
                   (-1, -1), (-1, 1), (1, -1), (1, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < img_grad.shape[1] and 0 <= ny < img_grad.shape[0] and labels[ny, nx] == 0:
            labels[ny, nx] = labels[y, x]
            heappush(heap, (img_grad[ny, nx], nx, ny))
end_time = time.time()
print("Time taken for watershed algorithm: ", end_time - start_time)
n = len(points)
colors = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
]
segmented_output = img_org.copy()
for i in range(1, n + 1):
    mask = labels == i
    if i < len(colors):
        segmented_output[mask] = 0.5 * segmented_output[mask] + 0.5 * np.array(colors[i])
    else:
        segmented_output[mask] = 0.5 * segmented_output[mask] + (0.5*np.random.randint(256), 0.5*np.random.randint(256), 0.5*np.random.randint(256))
cv2.imshow(f"mask", segmented_output)
cv2.waitKey(0)
