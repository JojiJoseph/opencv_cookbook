import kornia as K
import kornia.feature as KF
import cv2
import matplotlib.pyplot as plt
from kornia_moons.viz import draw_LAF_matches
import torch

cv2.samples.addSamplesDataSearchPath("../../test_images")

img0 = K.io.load_image(cv2.samples.findFile("graf1.png"), K.io.ImageLoadType.RGB32, device="cpu")[None, ...]
img1 = K.io.load_image(cv2.samples.findFile("graf3.png"), K.io.ImageLoadType.RGB32, device="cpu")[None, ...]

# img0 = K.geometry.resize(img0,(600,375))
print(K.color.rgb_to_grayscale(img0).shape)
matcher = KF.loftr.LoFTR()
# img0 = K.geometry.resize(img0, (480, 640), antialias=True)
# img1 = K.geometry.resize(img1, (480, 640), antialias=True)

inp = {"image0": K.color.rgb_to_grayscale(img0), "image1": K.color.rgb_to_grayscale(img1)}

with torch.inference_mode():
    out = matcher(inp)

mkpts0 = out["keypoints0"].cpu().numpy()
mkpts1 = out["keypoints1"].cpu().numpy()
Fm, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
inliers = inliers > 0
inliers[out["confidence"].cpu().numpy()<0.99] = 0

print(out.keys())

draw_LAF_matches(
    KF.laf_from_center_scale_ori(
        torch.from_numpy(mkpts0).view(1, -1, 2),
        torch.ones(mkpts0.shape[0]).view(1, -1, 1, 1),
        torch.ones(mkpts0.shape[0]).view(1, -1, 1),
    ),
    KF.laf_from_center_scale_ori(
        torch.from_numpy(mkpts1).view(1, -1, 2),
        torch.ones(mkpts1.shape[0]).view(1, -1, 1, 1),
        torch.ones(mkpts1.shape[0]).view(1, -1, 1),
    ),
    torch.arange(mkpts0.shape[0]).view(-1, 1).repeat(1, 2),
    K.tensor_to_image(img0),
    K.tensor_to_image(img1),
    inliers,
    draw_dict={"inlier_color": (0.2, 1, 0.2), "tentative_color": None, "feature_color": (0.2, 0.5, 1), "vertical": False},
)
plt.show()