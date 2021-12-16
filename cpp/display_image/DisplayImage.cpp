#include <stdio.h>
#include <opencv2/opencv.hpp>


int main() {
    cv::Mat img = cv::imread("../../test_images/lena.jpg");
    cv::imshow("Display Image", img);
    cv::waitKey();
}