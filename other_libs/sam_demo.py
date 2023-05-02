import cv2
import numpy as np
cv2.samples.addSamplesDataSearchPath("../test_images")

img_org = cv2.imread(cv2.samples.findFile("fruits.jpg"))

r = cv2.selectROI("Select a bounding box", img_org)

#exit()
from segment_anything import SamPredictor, sam_model_registry
sam = sam_model_registry["vit_b"](checkpoint="./sam_vit_b_01ec64.pth")
predictor = SamPredictor(sam)

predictor.set_image(img_org[:,:,::-1])
masks, _, _ = predictor.predict(box=np.array([r[0],r[1], r[0]+r[2], r[1]+r[3]]))
print(masks.min(),masks.max())
img_with_mask1 = img_org * masks[0].reshape(img_org.shape[0], img_org.shape[1],1)
cv2.imshow("img_with_mask1", img_with_mask1)
img_with_mask2 = img_org * masks[1].reshape(img_org.shape[0], img_org.shape[1],1)
cv2.imshow("img_with_mask2", img_with_mask2)
img_with_mask3 = img_org * masks[2].reshape(img_org.shape[0], img_org.shape[1],1)
cv2.imshow("img_with_mask3", img_with_mask3)
cv2.waitKey()
print(masks.shape)
