import torchvision
import cv2
import torchvision.transforms as T
import numpy as np

# Segmentation Model
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

COLORS = np.array([
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255,255,0],
    [0,255,255],
    [255,0,255],
    [128, 0, 0],
    [0, 128, 0],
    [0, 0, 128],
])


def segment_image(image, thresh=0.8):
    # Input is a cv2 image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = T.ToTensor()(image)
    out = model([image_tensor])[0]
    indices = (out["scores"] > thresh).detach().numpy()
    scores = out["scores"][indices].detach().numpy()
    labels = out["labels"][indices].detach().numpy()
    bounding_boxes = out["boxes"][indices].detach().numpy().astype(int)
    masks = out["masks"][indices].detach().numpy()
    return labels, scores, bounding_boxes, masks


def draw_segmentation(image, labels, bounding_boxes, masks, mask_thresh=0.8):
    img_out = image.copy()
    for mask in masks:
        mask = mask[0]
        color = COLORS[np.random.randint(len(COLORS))]
        color_img = np.ones_like(img_out, np.uint8)
        color_img[mask > mask_thresh] = color
        img_out = cv2.addWeighted(img_out, 1, color_img, 0.5, 0)
    for box, label in zip(bounding_boxes, labels):
        label = COCO_INSTANCE_CATEGORY_NAMES[label]
        cv2.rectangle(img_out, np.int32(
            box[:2]), np.int32(box[2:]), (255, 0, 0), 2)
        cv2.putText(img_out, label, np.int32(
            box[:2]) + [0, -10], cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 0), 2)
    return img_out



img = cv2.imread("./andrew-s-ouo1hbizWwo-unsplash.jpg") # https://unsplash.com/photos/ouo1hbizWwo
labels, scores, bounding_boxes, masks = segment_image(img, 0.7)
img_out = draw_segmentation(img, labels, bounding_boxes, masks, mask_thresh=0.3)
cv2.imshow("Original", img)
cv2.imshow("Segmented", img_out)
cv2.waitKey()
