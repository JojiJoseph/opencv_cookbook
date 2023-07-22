# pip install kornia
# pip install kornia_rs
import cv2
import kornia as K
import matplotlib.pyplot as plt

cv2.samples.addSamplesDataSearchPath("../../test_images")

img = K.io.load_image(cv2.samples.findFile("lena.jpg"), K.io.ImageLoadType.RGB32, device="cpu")
print("#1 type(img): ", type(img), " img.shape: ", img.shape)
img = K.tensor_to_image(img) # Change type and order of channels from CHW to HWC
print("#2 type(img): ", type(img), " img.shape: ", img.shape)
print(img.min(), img.max()) # Output range depends on the desired_type

plt.imshow(img)
plt.axis("off")
plt.show()