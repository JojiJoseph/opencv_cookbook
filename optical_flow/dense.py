import cv2
import numpy as np


cv2.samples.addSamplesDataSearchPath("../test_images")

video_path = cv2.samples.findFile("vtest.avi")

prev_frame = None
cap = cv2.VideoCapture(video_path)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Input", frame)
    if prev_frame is not None:
        prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prev_frame_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, angle = cv2.cartToPolar(flow[:,:,0], flow[:,:,1], angleInDegrees=True)
        mag = cv2.normalize(mag,None, 0, 1, cv2.NORM_MINMAX)
        angle = angle/(2*255)
        hsv = np.zeros(frame.shape)
        hsv[:,:,0] = angle
        hsv[:,:,1] = 1.
        hsv[:,:,2] = mag
        hsv = np.uint8(hsv*255)
        cv2.imshow("Dense optical flow", cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
    key = cv2.waitKey(1) & 0xFF
    if key in [27, ord('q')]:
        break
    prev_frame = frame
