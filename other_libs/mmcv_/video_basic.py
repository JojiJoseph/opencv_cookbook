import mmcv
import cv2
cv2.samples.addSamplesDataSearchPath("../../test_images")

# Read operation
img = mmcv.imread(cv2.samples.findFile("stuff.jpg"))
reader = mmcv.VideoReader(cv2.samples.findFile("Megamind.avi"))

for frame in reader:
    mmcv.imshow(frame, "frame", round(1000/reader.fps/10))

reader._set_real_position(0) # Hack to start frame extraction from the beginning

reader.cvt2frames("./frames/")

# Generate video from frames
mmcv.frames2video("./frames/", "./frames.avi", fps=reader.fps)

# Convert video
mmcv.convert_video("./frames.avi", "./frames.mp4")
# TODO
# - [] Cut video
# - [] Join videos
