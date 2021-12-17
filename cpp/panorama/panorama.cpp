#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main()
{
    // Images are taken from https://github.com/kriyeng/kornia-stitcher
    Mat img_left = imread("./bryce_left_02.png");
    Mat img_right = imread("./bryce_right_02.png");
    
    Ptr<ORB> detector = ORB::create();

    vector<KeyPoint> kp1, kp2;
    Mat des1, des2;
    detector->detectAndCompute(img_left, noArray(), kp1, des1);
    detector->detectAndCompute(img_right, noArray(), kp2, des2);

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE_HAMMING);
    vector<DMatch> matches;
    matcher->match(des1, des2, matches);
    sort(matches.begin(), matches.end(), [](DMatch one, DMatch two)
         { return one.distance < two.distance; });
    matches = vector<DMatch>(matches.begin(), matches.begin() + 10);
    Mat img_matches;
    drawMatches(img_left, kp1, img_right, kp2, matches, img_matches);
    vector<Point2d> pts1, pts2;
    for (DMatch match : matches)
    {
        pts1.push_back(kp1[match.queryIdx].pt);
        pts2.push_back(kp2[match.trainIdx].pt);
    }
    Mat h = findHomography(pts2, pts1, noArray(), RANSAC, 5.0);
    Mat img_out;

    warpPerspective(img_right, img_out, h, {img_left.size[1] + img_right.size[1], img_left.size[0] + img_right.size[0]});

    img_left.copyTo(img_out.rowRange(0, img_left.size[0]).colRange(0, img_left.size[1]));
    imshow("Panorama", img_out);
    waitKey();
}
