# Ref: https://youtu.be/mPCZLOVTEc4

import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
q
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    
    faces = face_cascade.detectMultiScale(frame)
    for x,y,w,h in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        eyes = eye_cascade.detectMultiScale(frame[y:y+h,x:x+w])
        for xe, ye, we, he in eyes:
            cv2.rectangle(frame,(x+xe,y+ye),(x+xe+we,y+ye+he),(0,0,255),2)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()