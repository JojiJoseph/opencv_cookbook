#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
int main()
{
    QRCodeDetector detector;
    Mat img = imread("./qr.png");
    std::vector<Point> points;
    std::vector<std::vector<Point> > wrapper;
    std::string value = detector.detectAndDecode(img, points);
    wrapper.push_back(points);
    drawContours(img,wrapper, -1, {255,0,0}, 2);
    putText(img, value, {10,15}, FONT_HERSHEY_COMPLEX_SMALL,1,{0,0,255},2);
    imshow("QR", img);
    waitKey();
    std::cout << "Detected value: " << value << std::endl;
}
