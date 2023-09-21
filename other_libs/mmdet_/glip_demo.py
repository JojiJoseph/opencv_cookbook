from io import BytesIO
from mmdet.apis import init_detector, inference_detector
import cv2
import numpy as np
import PIL.Image as Image
import requests


def load(url):
    """
    Given an url of an image, downloads the image and
    returns a PIL image
    """
    response = requests.get(url)
    pil_image = Image.open(BytesIO(response.content)).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image


# Please do clone mmdetection and download the checkpoint file before running the code
config_file = 'mmdetection/configs/glip/glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365.py'
checkpoint_file = 'glip_tiny_a_mmdet-b3654169.pth'

cv2.samples.addSamplesDataSearchPath("../../test_images")
# image = cv2.imread(cv2.samples.findFile("basketball1.png"))
image = load('http://farm4.staticflickr.com/3693/9472793441_b7822c00de_z.jpg')
image = np.ascontiguousarray(image)


model = init_detector(config_file, checkpoint_file, device='cpu')
output = inference_detector(model, image, text_prompt="cupboard. chair. sofa")

if len(output.pred_instances.bboxes) == 0:
    print("No detection")
    exit()
print(output.pred_instances)
for bbox, score, label in zip(output.pred_instances.bboxes, output.pred_instances.scores, output.pred_instances.label_names):
    print(bbox, score)
    bbox = bbox.cpu().int().tolist()
    if score > 0.5:
        cv2.rectangle(image, (bbox[0], bbox[1]),
                      (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(image, label, (bbox[0], bbox[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.imshow("image", image)
cv2.waitKey()
print(output.pred_instances.keys())
print(output.pred_instances.label_names)
