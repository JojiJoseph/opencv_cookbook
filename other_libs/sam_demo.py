import time
import cv2
import numpy as np
from cached_path import cached_path
from segment_anything import SamPredictor, sam_model_registry

cv2.samples.addSamplesDataSearchPath("../test_images")

checkpoints = {
    "vit_b":"https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "vit_l":"https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_h":"https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
}

checkpoint_path = cached_path(checkpoints["vit_b"])

sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path).to('cuda')

predictor = SamPredictor(sam)
img_org = cv2.imread(cv2.samples.findFile("fruits.jpg"))
r = cv2.selectROI("Select a bounding box", img_org) # Output is of the form x,y,w,h


start_time = time.time()
predictor.set_image(img_org, image_format="BGR")
masks, scores, logits = predictor.predict(box=np.array([r[0],r[1], r[0]+r[2], r[1]+r[3]]))
end_time = time.time()
print("Time taken for prediction: ", end_time - start_time)
masks = masks[np.argsort(scores)][::-1]
scores = np.sort(scores)[::-1]
img_with_mask1 = img_org * masks[0].reshape(img_org.shape[0], img_org.shape[1],1)
cv2.imshow(f"img_with_mask1, score:{np.round(scores[0],3)}", img_with_mask1)
img_with_mask2 = img_org * masks[1].reshape(img_org.shape[0], img_org.shape[1],1)
cv2.imshow(f"img_with_mask2, score:{np.round(scores[1],3)}", img_with_mask2)
img_with_mask3 = img_org * masks[2].reshape(img_org.shape[0], img_org.shape[1],1)
cv2.imshow(f"img_with_mask3, score:{np.round(scores[2],3)}", img_with_mask3)
cv2.waitKey()

