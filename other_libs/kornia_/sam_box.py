import kornia as K
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from kornia.geometry.boxes import Boxes
from kornia.geometry.keypoints import Keypoints

from kornia.utils.helpers import get_cuda_device_if_available


from kornia.contrib.image_prompter import ImagePrompter
from kornia.contrib.models.sam import SamConfig
from kornia.contrib.models.structures import SegmentationResults

device = get_cuda_device_if_available()

model_type = "vit_b"
checkpoint = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"


cv2.samples.addSamplesDataSearchPath("../../test_images")

img_tensor = K.io.load_image(cv2.samples.findFile("stuff.jpg"), K.io.ImageLoadType.RGB32, device=device)
img = K.utils.tensor_to_image(img_tensor)
img = np.clip(img*255, 0, 255).astype(np.uint8)


r = cv2.selectROI("Select a bounding box", img[...,::-1])


# Setting up a SamConfig with the model type and checkpoint desired
config = SamConfig(model_type, checkpoint)

# Initialize the ImagePrompter
prompter = ImagePrompter(config, device=device)
prompter.set_image(img_tensor)
boxes_tensor = torch.tensor([[[r[0], r[1], r[0]+r[2], r[1]+r[3]]]], device=device, dtype=torch.float32)
boxes = Boxes.from_tensor(boxes_tensor, mode="xyxy")

results: SegmentationResults = prompter.predict(boxes=boxes)

mask_idx = results.scores[0].argmax(dim=-1)
mask = results.logits[0][mask_idx] > results.mask_threshold
mask = (mask.cpu().numpy()*255).astype(np.uint8)

mask = mask[...,None] & [255,0,0]
mask = mask.astype(np.uint8)
print(img.shape, mask.shape, mask.dtype)
mask = cv2.resize(mask, (max(img.shape), max(img.shape)))
if img.shape[0] < img.shape[1]:
    mask = mask[:img.shape[0]]
else:
    mask = mask[:,:img.shape[1]]
img_out = cv2.addWeighted(img, 0.5, mask, 0.5,1)
cv2.imshow("mask", mask[...,::-1])
cv2.imshow("img_out", img_out[...,::-1])

cv2.waitKey()