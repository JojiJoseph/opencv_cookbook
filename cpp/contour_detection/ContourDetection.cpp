#include <opencv2/opencv.hpp>
#include <iostream>

int main()
{
    cv::samples::addSamplesDataSearchPath("../../test_images");
    cv::Mat img_org = cv::imread(cv::samples::findFile("pca_test1.jpg"));
    cv::Mat img, img_out;
    cv::blur(img_org, img, {5, 5});
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    cv::threshold(img, img, 100, 255, cv::THRESH_BINARY);
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(img, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    cv::drawContours(img_org, contours, -1, {0, 0, 255}, 4);
    cv::imshow("Contour Detection", img_org);
    cv::waitKey();
    cv::destroyAllWindows();
}
