#include <surveillance.hpp>

const std::string currentDateTime()
{
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);

    strftime(buf, sizeof buf, "%Y-%m-%d %X", &tstruct);

    return buf;
}

void calcMeanImage(std::vector<cv::Mat> const &backgroundImages, cv::Mat &meanImage)
{
    CV_Assert(!backgroundImages.empty());

    double factor = 1 / static_cast<double>(backgroundImages.size());
    cv::Mat accumImage = cv::Mat::zeros(backgroundImages[0].size(), CV_64FC3);

    std::for_each(backgroundImages.begin(), backgroundImages.end(), [&accumImage, factor](cv::Mat const &item)
    {
            cv::Mat converted;
            item.convertTo(converted, CV_64FC3, factor);
            accumImage += converted;
    });

    accumImage.convertTo(meanImage, CV_8UC3);
    cv::cvtColor(meanImage, meanImage, CV_BGR2GRAY);

}

void initilizeBackground(const unsigned historySize, cv::Mat &background,
                         std::vector<cv::Mat> backgroundImages, cv::VideoCapture  &capture)
{
    while(backgroundImages.size() < historySize)
    {
        cv::Mat frame;
        capture >> frame;
        cv::imshow("video", frame);
        int key = cv::waitKey(30) & 0xFF;
        if(key == 27)
            break;
        if(key == 10)
            backgroundImages.push_back(frame);
    }

    CV_Assert(backgroundImages.size() == historySize);

    calcMeanImage(backgroundImages, background);
}
