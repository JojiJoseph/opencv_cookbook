import cv2
import numpy as np

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
facemark = cv2.face.createFacemarkLBF()
facemark.loadModel("./lbfmodel.yaml") # Thanks: https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml

def draw_landmarks(frame, landmarks):
    landmarks = np.array(landmarks).astype(np.int32)
    # Drawing face contour
    cv2.polylines(frame, landmarks[None,:17,:], False, (0,255,255))
    # Right eybrow
    cv2.polylines(frame, landmarks[None,17:22,:], False, (0,255,255))
    # Left eybrow
    cv2.polylines(frame, landmarks[None,22:27,:], False, (0,255,255))
    # Nose vertical
    cv2.polylines(frame, landmarks[None,27:31,:], False, (0,255,255))
    # Nose bottom
    cv2.polylines(frame, landmarks[None,31:36,:], False, (0,255,255))
    # Right eye
    cv2.polylines(frame, landmarks[None,36:42,:], True, (0,255,255))
    # Left eye
    cv2.polylines(frame, landmarks[None,42:48,:], True, (0,255,255))
    # Lip outer contour
    cv2.polylines(frame, landmarks[None,48:60,:], True, (0,255,255))
    # Lip inner contour
    cv2.polylines(frame, landmarks[None,60:68,:], True, (0,255,255))

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(frame_gray,1.25,5)

    if len(faces):
        success, landmarks = facemark.fit(frame, faces)
        if success:
            for face_landmarks in landmarks:
                draw_landmarks(frame, face_landmarks[0])
        for x,y,w,h in faces:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255))
    
    cv2.imshow("Facial Landmarks", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()