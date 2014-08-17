#include <opencv2/opencv.hpp>

#include "components.hpp"

int main(int argc, char **argv)
{
    if(argc == 1) return std::cerr << "nothing to do, please specify binary images" << std::endl, -1;

    std::string imagePath = argv[1];

    cv::Mat binaryImage, originalImage = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);

    if(originalImage.empty()) return std::cerr << "cannot open " << imagePath << std::endl, -1;

    cv::threshold(originalImage, binaryImage, 0, 255, CV_THRESH_OTSU);

    cv::imshow("binary image", binaryImage);

    std::vector< cv::ConnectedComponent > connectedComponents;

    cv::findConnectedComponents(binaryImage, connectedComponents, cv::fourConnected);

    cv::waitKey(0);

    return 0;
}
