from re import sub
import cv2
import numpy as np
from torch import seed
from heapq import heappush, heappop, heapify
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str, default=None) # TODO: Implement this
parser.add_argument("--edge-detector", type=str, choices=["sobel", "laplacian"], default="laplacian")
args = parser.parse_args()

cv2.samples.addSamplesDataSearchPath("../test_images")

img_org = cv2.imread(cv2.samples.findFile("./fruits.jpg"))

img_gray = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)

img_laplacian = cv2.Laplacian(img_gray, cv2.CV_8U)
img_laplacian = cv2.equalizeHist(img_laplacian)
img_sobel = cv2.Sobel(img_gray, cv2.CV_8U, 1, 1)
img_sobel = cv2.equalizeHist(img_sobel)

if args.edge_detector == "laplacian":
    graph = 255 - img_laplacian
else:
    graph = 255 - img_sobel

# mouse listener to find two point in the image
points = []
subpoints = []


def shortest_path_dijkstra(graph: np.ndarray, src, dst):
    # graph = 255 - graph.astype(np.float32)
    # graph = 1 - graph/255.
    graph = graph.astype(np.float32) #+ 1.0
    heap = []
    visited  = set()
    heappush(heap, (0, src, src))
    path = {}
    dist = {}
    nodes_visited = []
    
    while heap:
        d, node, parent = heappop(heap)
        nodes_visited.append(node)
        visited.add(node)
        path[node] = parent
        if node == dst:
            break
        x, y = node
        for dx, dy in [(0,1), (0,-1), (1,0), (-1,0), (-1,-1), (1,1), (1,-1), (-1,1)]:
            if 0 <=y+ dy < graph.shape[0] and 0 <= x+dx < graph.shape[1]:
                edge_weight = 1 + abs(graph[y + dy,x + dx] - graph[y, x]) + graph[y + dy,x + dx]
                if ((x + dx, node[1] + dy) not in dist or dist[(node[0] + dx, node[1] + dy)] > edge_weight + d + 0*graph[y + dy,x + dx]) and (node[0] + dx, node[1] + dy) not in visited:
                    dist[(node[0] + dx, node[1] + dy)] = d + edge_weight + 0*graph[y+dy,x+dx]
                    heappush(heap, (d + edge_weight + 0*graph[y+dy,x+dx], (x+dx,y+dy), node))
    node = dst
    points = []
    while path[node] != node:
        points.append(list(node))
        node = path[node]
    return points[::-1], nodes_visited

def shortest_path_astar(graph: np.ndarray, src, dst):
    # graph = 255 - graph.astype(np.float32)
    # graph = 1 - graph/255.
    graph = graph.astype(np.float32) #+ 1.0
    # print(graph.max(), graph.min())
    # exit()
    # dst_np = np.array(dst)
    heap = []
    visited  = set()
    heappush(heap, (np.linalg.norm(list(dst)), src, src))
    path = {}
    dist = {src:0}
    nodes_visited = []
    
    while heap:
        d, node, parent = heappop(heap)
        nodes_visited.append(node)
        visited.add(node)
        path[node] = parent
        if node == dst:
            break
        x, y = node
        for dx, dy in [(0,1), (0,-1), (1,0), (-1,0), (-1,-1), (1,1), (1,-1), (-1,1)]:
            if 0 <=y+ dy < graph.shape[0] and 0 <= x+dx < graph.shape[1]:
                edge_weight = 1 + abs(graph[y + dy,x + dx] - graph[y, x]) + graph[y + dy,x + dx]
                if ((x + dx, y + dy) not in dist or dist[(x + dx, y + dy)] > edge_weight + dist[(x,y)] + 0*graph[y + dy,x + dx]) and (node[0] + dx, node[1] + dy) not in visited:
                    # dist[(x + dx, y + dy)] = dist[(x,y)] + 0 + graph[y+dy,x+dx]
                    dist[(x + dx, y + dy)] = dist[(x,y)] + edge_weight #+ graph[y+dy,x+dx]
                    heappush(heap, (dist[(x+dx,y+dy)] + 1*np.linalg.norm([dst[0]-(x+dx), dst[1]-(y+dy)], ord=1), (x+dx,y+dy), node))
    node = dst
    points = []
    while path[node] != node:
        points.append(list(node))
        node = path[node]
    return points[::-1], nodes_visited

        
clicked = False
nodes_visited = set()
timings = []

# Video writer for intellignet scissors animation
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter('intelligent_scissors.avi', fourcc, 20.0, (img_org.shape[1], img_org.shape[0]))

def mouse_event(event, x, y, flags, param):
    global clicked, subpoints, points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) > 0:
            start_time = time.time()
            subpoints_extra, nodes_visited_extra = shortest_path_astar(graph, src=tuple(points[-1]), dst=(x,y))
            # img_out = img_org.copy()
            img_out = img_laplacian.copy()
            
            color_delta = 255 / len(nodes_visited_extra)
            # steps = 1 + len(nodes_visited_extra) // 2
            for i in range(0, len(nodes_visited_extra), 1000):
                # img_out = img_org.copy()
                img_out = img_laplacian.copy()
                img_out = cv2.cvtColor(img_laplacian, cv2.COLOR_GRAY2BGR)
                mask = np.zeros_like(img_out)
                for idx, point in enumerate(nodes_visited_extra[:i]):
                    color_b  = 255 - color_delta * idx
                    color_g = 255 - color_delta * (len(nodes_visited_extra) - idx)
                    # print(color_b, color_g)
                    color_b, color_g = int(color_b), int(color_g)
                    cv2.circle(mask, tuple(point), 1, (color_b,color_g,0), thickness=-1)
                img_out = cv2.addWeighted(img_out, 1, mask, 0.6, 0)
                for point in subpoints:
                    cv2.circle(img_out, tuple(point), 1, (0,255,255), thickness=-1)
                for point in points:
                    cv2.circle(img_out, tuple(point), 5, (0,0,255), thickness=-1)
                cv2.circle(img_out, (x,y), 5, (0,0,255), thickness=-1)
                # cv2.imshow("Animation", img_out)
                # cv2.waitKey(1)
                video_writer.write(img_out)
            # steps = 1 + len(subpoints_extra) // 2
            for i in range(0, len(subpoints_extra), 20):
                # img_out = img_org.copy()
                img_out = img_laplacian.copy()
                img_out = cv2.cvtColor(img_laplacian, cv2.COLOR_GRAY2BGR)
                mask = np.zeros_like(img_out)
                for idx, point in enumerate(nodes_visited_extra):
                    color_b  = 255 - color_delta * idx
                    color_g = 255 - color_delta * (len(nodes_visited_extra) - idx)
                    # print(color_b, color_g)
                    color_b, color_g = int(color_b), int(color_g)
                    cv2.circle(mask, tuple(point), 1, (color_b,color_g,0), thickness=-1)
                img_out = cv2.addWeighted(img_out, 1, mask, 0.6, 0)
                for point in subpoints_extra[::-1][:i]:
                    cv2.circle(img_out, tuple(point), 1, (0,255,255), thickness=-1)
                for point in subpoints:
                    cv2.circle(img_out, tuple(point), 1, (0,255,255), thickness=-1)
                for point in points:
                    cv2.circle(img_out, tuple(point), 5, (0,0,255), thickness=-1)
                cv2.circle(img_out, (x,y), 5, (0,0,255), thickness=-1)
                # cv2.imshow("Animation", img_out)
                # cv2.waitKey(1)
                # cv2.destroyAllWindows()
                video_writer.write(img_out)

            # for point in points:
                # cv2.circle(img_out, tuple(point), 5, (0,0,255), thickness=-1)
            subpoints.extend(subpoints_extra)
            nodes_visited.update(nodes_visited_extra)
            end_time = time.time()
            timings.append(end_time - start_time)
        points.append([x,y])

cv2.namedWindow("Intelligent Scissors", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Intelligent Scissors", mouse_event)
while True:
    img_out  = img_org.copy()
    # for x,y in nodes_visited:
    #     cv2.circle(img_out, (x,y), 1, (255,0,0), thickness=-1) # Explored nodes
    for x, y in subpoints:
        cv2.circle(img_out, (x,y), 1, (0,255,255), thickness=-1)
    for x, y in points:
        cv2.circle(img_out, (x,y), 5, (0,0,255), thickness=-1)
    cv2.imshow("Intelligent Scissors", img_out)
    key = cv2.waitKey(1) & 0xFF
    if key in [27, ord('q')]:
        break
for _ in range(20):
    for point in subpoints:
        cv2.circle(img_out, tuple(point), 1, (0,255,255), thickness=-1)
    for point in points:
        cv2.circle(img_out, tuple(point), 5, (0,0,255), thickness=-1)
    video_writer.write(img_out)
print("Average time taken for processing 1 curve", np.mean(timings))
print("Maximum time taken for processing 1 curve", np.max(timings))
print("Minimum time taken for processing 1 curve", np.min(timings))
video_writer.release()