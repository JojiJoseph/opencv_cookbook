#include <opencv2/opencv.hpp>

int main()
{
    cv::samples::addSamplesDataSearchPath("../../test_images");
    cv::String img_path = cv::samples::findFile("fruits.jpg");
    cv::Mat img = cv::imread(img_path), img_hsv;
    cv::cvtColor(img, img_hsv, cv::COLOR_BGR2HSV);

    cv::namedWindow("Image");

    int low_h = 0, low_s = 0, low_v = 0;
    int high_h = 179, high_s = 255, high_v = 255;

    cv::createTrackbar("low_h", "Image", &low_h, 179);
    cv::createTrackbar("low_s", "Image", &low_s, 255);
    cv::createTrackbar("low_v", "Image", &low_v, 255);
    cv::createTrackbar("high_h", "Image", &high_h, 179);
    cv::createTrackbar("high_s", "Image", &high_s, 255);
    cv::createTrackbar("high_v", "Image", &high_v, 255);

    while (1)
    {
        std::vector<int> lowhsv = {low_h, low_s, low_v};
        std::vector<int> highhsv = {high_h, high_s, high_v};
        cv::Mat mask, segmented_fruits;

        cv::inRange(img_hsv, lowhsv, highhsv, mask);

        cv::bitwise_and(img, img, segmented_fruits, mask);
        cv::imshow("Image", segmented_fruits);

        if (cv::waitKey(1) == 'q')
            break;
    }
    return 0;
}