#include <opencv2/opencv.hpp>

int main(int, char** )
{
    cv::Mat image(480, 640, CV_8UC3, cv::Scalar::all(0));

    cv::putText(image, "Привет мир!", cv::Point(200,200), cv::FONT_HERSHEY_COMPLEX, 1., cv::Scalar(0,255,0));

    cv::imshow("image", image);
    cv::waitKey();

    return EXIT_SUCCESS;

}
