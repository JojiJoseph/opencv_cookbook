from re import sub
import cv2
import numpy as np
from torch import seed
from heapq import heappush, heappop, heapify
import argparse

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


graph = 255 - img_laplacian

# mouse listener to find two point in the image
points = []
subpoints = []


def shortest_path_dijkstra(graph: np.ndarray, src, dst):
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
                if ((x + dx, node[1] + dy) not in dist or dist[(node[0] + dx, node[1] + dy)] > 1 + d + graph[y + dy,x + dx]) and (node[0] + dx, node[1] + dy) not in visited:
                    dist[(node[0] + dx, node[1] + dy)] = d + 1 + graph[y+dy,x+dx]
                    heappush(heap, (d + 1 + graph[y+dy,x+dx], (x+dx,y+dy), node))
    node = dst
    points = []
    while path[node] != node:
        points.append(list(node))
        node = path[node]
    return points[::-1], nodes_visited

def shortest_path_astar(graph: np.ndarray, src, dst):
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
                if ((x + dx, y + dy) not in dist or dist[(x + dx, y + dy)] > 1 + dist[(x,y)] + graph[y + dy,x + dx]) and (node[0] + dx, node[1] + dy) not in visited:
                    dist[(x + dx, y + dy)] = dist[(x,y)] + 1 + graph[y+dy,x+dx]
                    heappush(heap, (dist[(x+dx,y+dy)] + np.linalg.norm([dst[0]-(x+dx), dst[1]-(y+dy)]), (x+dx,y+dy), node))
    node = dst
    points = []
    while path[node] != node:
        points.append(list(node))
        node = path[node]
    return points[::-1], nodes_visited

        
clicked = False
nodes_visited = set()
def mouse_event(event, x, y, flags, param):
    global clicked, subpoints, points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) > 0:
            subpoints_extra, nodes_visited_extra = shortest_path_dijkstra(graph, src=tuple(points[-1]), dst=(x,y))
            subpoints.extend(subpoints_extra)
            nodes_visited.update(nodes_visited_extra)
        points.append([x,y])

cv2.namedWindow("Graph")
cv2.setMouseCallback("Graph", mouse_event)

while True:
    img_out  = img_org.copy()
    # for x,y in nodes_visited:
    #     cv2.circle(img_out, (x,y), 1, (255,0,0), thickness=-1) # Explored nodes
    for x, y in subpoints:
        cv2.circle(img_out, (x,y), 1, (0,255,255), thickness=-1)
    for x, y in points:
        cv2.circle(img_out, (x,y), 5, (0,0,255), thickness=-1)
    cv2.imshow("Graph", img_out)
    key = cv2.waitKey(1) & 0xFF
    if key in [27, ord('q')]:
        break
#
