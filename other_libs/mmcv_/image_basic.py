"""Basic image processing operations with mmcv."""

import mmcv
import cv2
import numpy as np

cv2.samples.addSamplesDataSearchPath("../../test_images")

# Read operation
img = mmcv.imread(cv2.samples.findFile("stuff.jpg"))
mmcv.imshow(img, "img")

# Write operation
mmcv.imwrite(img, "img_out.jpg")

# Read from bytes
with open(cv2.samples.findFile("stuff.jpg"), "rb") as f:
    img_bytes = f.read()
img = mmcv.imfrombytes(img_bytes)
mmcv.imshow(img, "img from bytes")

# Color conversion
img_gray = mmcv.bgr2gray(img)
mmcv.imshow(img_gray, "img_gray")

# Resize
img_resize = mmcv.imresize(img, (img.shape[1]//2, img.shape[0]//2))
mmcv.imshow(img_resize, "img_resize")

img_rescale = mmcv.imrescale(img, 2)
mmcv.imshow(img_rescale, "img_rescale")

img_rescale_keep_aspect, scale = mmcv.imrescale(img, (512, 512), return_scale=True)
mmcv.imshow(img_rescale_keep_aspect, "img_rescale_keep_aspect")
print(scale, img.shape, img_rescale_keep_aspect.shape)

# Rotate
img_rotate = mmcv.imrotate(img, 45)
mmcv.imshow(img_rotate, "img_rotate")

img_rotate_pivot_top_left = mmcv.imrotate(img, 45,center=(0,0))
mmcv.imshow(img_rotate_pivot_top_left, "img_rotate_pivot_top_left")

img_rotate_auto_bound = mmcv.imrotate(img, 45, auto_bound=True)
mmcv.imshow(img_rotate_auto_bound, "img_rotate_auto_bound")

# Crop
bboxes = np.array([[0,0,100,100], [100,100,200,200]])

img_crops = mmcv.imcrop(img, bboxes)
for i, img_crop in enumerate(img_crops):
    mmcv.imshow(img_crop, f"img_crop_{i}")

img_equalize = mmcv.imequalize(img)
mmcv.imshow(img_equalize, "img_equalize")

img_pad_constant = mmcv.impad(img, shape=(2000, 2000), pad_val=(255,0,0), padding_mode="constant")
mmcv.imshow(img_pad_constant, "img_pad_constant")


img_pad_edge = mmcv.impad(img, shape=(2000, 2000), pad_val=(255,0,0), padding_mode="edge")
mmcv.imshow(img_pad_edge, "img_pad_edge")

img_pad_reflect = mmcv.impad(img, shape=(2000, 2000), pad_val=(255,0,0), padding_mode="reflect")
mmcv.imshow(img_pad_reflect, "img_pad_reflect")

img_pad_symmetric = mmcv.impad(img, shape=(2000, 2000), pad_val=(255,0,0), padding_mode="symmetric")
mmcv.imshow(img_pad_symmetric, "img_pad_symmetric")

img_pad_both = mmcv.impad(img, padding=(700-img.shape[1], 700-img.shape[0]), pad_val=(255,0,0))
mmcv.imshow(img_pad_both, "img_pad_both")
