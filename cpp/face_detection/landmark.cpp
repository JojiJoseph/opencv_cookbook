// import cv2
// import numpy as np
#include <opencv2/opencv.hpp>
#include <opencv2/face/facemarkLBF.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void draw_landmarks(Mat frame, vector<Point2f> landmarks)
{
    vector<Point2i> landmarks_int(landmarks.begin(), landmarks.end());
    // Face contour
    vector<Point2i> face_contour(landmarks_int.begin(), landmarks_int.begin() + 17);
    polylines(frame, face_contour, false, {0, 255, 255});
    // Right eyebrow
    vector<Point2i> right_eyebrow(landmarks_int.begin() + 17, landmarks_int.begin() + 22);
    polylines(frame, right_eyebrow, false, {0, 255, 255});
    // Left eybrow
    vector<Point2i> left_eyebrow(landmarks_int.begin() + 22, landmarks_int.begin() + 27);
    polylines(frame, left_eyebrow, false, {0, 255, 255});
    // Nose vertical
    vector<Point2i> nose_vert(landmarks_int.begin() + 27, landmarks_int.begin() + 31);
    polylines(frame, nose_vert, false, {0, 255, 255});
    // Nose bottom
    vector<Point2i> nose_bottom(landmarks_int.begin() + 31, landmarks_int.begin() + 36);
    polylines(frame, nose_bottom, false, {0, 255, 255});
    // Right eye
    vector<Point2i> right_eye(landmarks_int.begin() + 36, landmarks_int.begin() + 42);
    polylines(frame, right_eye, true, {0, 255, 255});
    // Left eye
    vector<Point2i> left_eye(landmarks_int.begin() + 42, landmarks_int.begin() + 48);
    polylines(frame, left_eye, true, {0, 255, 255});
    // Lip outer contour
    vector<Point2i> lip_outer_contour(landmarks_int.begin() + 48, landmarks_int.begin() + 60);
    polylines(frame, lip_outer_contour, true, {0, 255, 255});
    // Lip inner contour
    vector<Point2i> lip_inner_contour(landmarks_int.begin() + 60, landmarks_int.end());
    polylines(frame, lip_inner_contour, true, {0, 255, 255});
}

int main()
{
    VideoCapture cap(0);
    CascadeClassifier face_detector;
    Ptr<cv::face::FacemarkLBF> facemark = cv::face::FacemarkLBF::create();
    facemark->loadModel("./lbfmodel.yaml"); // Thanks: https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml
    face_detector.load("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml");
    while (1)
    {
        Mat frame;
        cap.read(frame);
        Mat frame_gray;
        cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
        vector<Rect> faces;
        face_detector.detectMultiScale(frame_gray, faces, 1.25, 5);
        if (faces.size())
        {
            vector<vector<Point2f>> landmarks;
            facemark->fit(frame, faces, landmarks);
            for (vector<Point2f> face_landmarks : landmarks)
            {
                draw_landmarks(frame, face_landmarks);
            }
        }
        imshow("Output", frame);
        char key = waitKey(1) & 0xFF;
        if (key == 'q')
        {
            break;
        }
    }
    cap.release();
    destroyAllWindows();
    return 0;
}
