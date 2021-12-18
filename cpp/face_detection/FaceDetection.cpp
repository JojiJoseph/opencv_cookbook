#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;

int main()
{
    CascadeClassifier face_cascade, eye_cascade;
    face_cascade.load("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml");
    eye_cascade.load("/usr/share/opencv4/haarcascades/haarcascade_eye.xml");
    VideoCapture cap(0);
    while (1)
    {
        Mat frame;
        cap.read(frame);
        std::vector<Rect> rects;
        face_cascade.detectMultiScale(frame, rects);
        for (Rect bbox : rects)
        {
            rectangle(frame, bbox, {0, 0, 255}, 2);
            std::vector<Rect> eyes;
            eye_cascade.detectMultiScale(frame.rowRange(bbox.y, bbox.y + bbox.height).colRange(bbox.x, bbox.x + bbox.width), eyes);

            for (Rect bbox_eye : eyes)
            {
                bbox_eye = Rect2i(bbox.x + bbox_eye.x, bbox.y + bbox_eye.y, bbox_eye.width, bbox_eye.height);
                rectangle(frame, bbox_eye, {255, 0, 0}, 2);
            }
        }

        imshow("", frame);
        char key = waitKey(1);
        if (key == 'q')
        {
            break;
        }
    }
    cap.release();
    destroyAllWindows();
    return 0;
}
