import time
import cv2
import numpy as np
from cached_path import cached_path
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator

cv2.samples.addSamplesDataSearchPath("../test_images")

checkpoints = {
    "vit_b":"https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "vit_l":"https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_h":"https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
}

checkpoint_path = cached_path(checkpoints["vit_b"])
sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path).to('cuda')

generator = SamAutomaticMaskGenerator(sam)

img_org = cv2.imread(cv2.samples.findFile("fruits.jpg"))
img_out = img_org.copy()
start_time = time.time()
output = generator.generate(img_org[...,::-1])
end_time = time.time()
print("time taken for generating all masks", end_time - start_time)

output.sort(key=lambda x:x['stability_score'])
for idx, mask_data in enumerate(output):
    x, y, w, h = mask_data['bbox']
    color = np.random.randint(50, 255,(3,))
    segment_colored = color * mask_data['segmentation'][...,None]
    segment_colored = segment_colored.astype(np.uint8)
    # cv2.imshow(f"mask {idx}", segment_colored) # Uncomment to see the mask
    img_out = cv2.addWeighted(img_out, 1, segment_colored, 0.5, 1)

cv2.imshow("Masks", img_out)
cv2.waitKey()
