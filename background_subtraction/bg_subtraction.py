import numpy as np
import cv2

cap = cv2.VideoCapture(0)

bg_subtractor = cv2.createBackgroundSubtractorMOG2()

while True:
    _, frame = cap.read()
    mask = bg_subtractor.apply(frame)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel=np.ones((5,5)))
    img_segmented = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.imshow("Mask", mask)
    cv2.imshow("Segmented", img_segmented)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()