import numpy as np
import cv2
from rodrigues  import rvec_to_R, R_to_rvec


def project_points(points, rvec, tvec, camera_matrix):
    R = rvec_to_R(rvec)
    # R = rvec
    camera_points = (R @ points.T).T + tvec.T
    image_points = (camera_matrix @ camera_points.T).T
    image_points /= image_points[:,[2]]
    return image_points[:,:-1]

def get_rot_matrix(roll, pitch, yaw):
    R_roll = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)],
        ])
    R_pitch = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)],
        ])
    R_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1],
        ])
    return R_yaw @ R_pitch @ R_roll


camera_matrix = np.array([
    [400, 0, 200],
    [0, 400., 200],
    [0, 0, 1]
])


tvec = np.array([0, 0, 2.]).reshape((3,1))

points = np.array([[-1,1,0],[1,1,0],[1,-1,0],[-1,-1.0,0]])

cv2.namedWindow("Project Points", cv2.WINDOW_NORMAL)
cv2.createTrackbar("Roll", "Project Points", 0, 359, lambda x:x)
cv2.createTrackbar("Pitch", "Project Points", 0, 359, lambda x:x)
cv2.createTrackbar("Yaw", "Project Points", 0, 359, lambda x:x)
cv2.createTrackbar("Distance", "Project Points", 4, 10, lambda x:x)


while True:
    img_left = np.zeros((400,400,3), np.uint8)
    img_right = np.zeros((400,400,3), np.uint8)

    roll = np.radians(cv2.getTrackbarPos("Roll", "Project Points"))
    pitch = np.radians(cv2.getTrackbarPos("Pitch", "Project Points"))
    yaw = np.radians(cv2.getTrackbarPos("Yaw", "Project Points"))
    tvec[-1][0] = cv2.getTrackbarPos("Distance", "Project Points")

    rvec,_ = cv2.Rodrigues(get_rot_matrix(roll, pitch, yaw))
    img_points,_ = cv2.projectPoints(points.reshape(-1,3), rvec,tvec,camera_matrix, None)
    img_points = img_points.astype(int)
    cv2.polylines(img_left, [img_points], True, (255,0,0), 2)
    cv2.putText(img_left, "OpenCV project points", (20,20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,255))

    # Custom project points
    # R = rvec_to_R(rvec)
    # rvec = R_to_rvec(R) # Fix this
    img_points = project_points(points.reshape(-1,3), rvec, tvec,camera_matrix)
    img_points = img_points.astype(int)
    # print(img_points.shape)
    cv2.polylines(img_right, [img_points], True, (255,0,0), 2)
    cv2.putText(img_right, "Custom project points", (20,20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,255))
    img = np.concatenate([img_left, img_right], axis=1)
    cv2.imshow("Project Points", img)
    key = cv2.waitKey(1000//30) & 0xFF
    if key in [27, ord('q')]:
        break