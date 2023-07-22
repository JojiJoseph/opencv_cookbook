# pip install kornia
# pip install kornia_rs
import cv2
import kornia as K
import matplotlib.pyplot as plt

cv2.samples.addSamplesDataSearchPath("../../test_images")

img = K.io.load_image(cv2.samples.findFile("lena.jpg"), K.io.ImageLoadType.RGB32, device="cpu")
img_gray = K.color.bgr_to_grayscale(img)

img = K.tensor_to_image(img) # Change type and order of channels from CHW to HWC
img_gray = K.tensor_to_image(img_gray) # Change type

plt.imshow(img)
plt.axis("off")
plt.title("RGB")
plt.show()

plt.imshow(img_gray, cmap="gray")
plt.axis("off")
plt.title("Grayscale")
plt.show()