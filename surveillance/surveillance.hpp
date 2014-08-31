#include <ctime>
#include <iomanip>

#include <opencv2/opencv.hpp>

//#define HIGH_DEF

const std::string currentDateTime();

void calcMeanImage(std::vector<cv::Mat> const &backgroundImages, cv::Mat &meanImage);

void initilizeBackground(const unsigned historySize, cv::Mat &background,
                         std::vector<cv::Mat> backgroundImages, cv::VideoCapture  &capture);
