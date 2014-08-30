#include <ctime>
#include <iomanip>

#include <opencv2/opencv.hpp>

#include <components.hpp>


//#define HIGH_DEF

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

int main(int, char** )
{
    const int alarmThreshhold = 17000;
    const double fps = 20.0;
    const int saveFrameCount = 150;
    const unsigned minArea = 10;
    double sum;
    int frameCount = 0;

#ifdef HIGH_DEF
    const int height = 720;
    const int width = 1280;
#else
    const int height = 480;
    const int width = 640;
#endif

    cv::VideoCapture capture(-1);
    capture.set(CV_CAP_PROP_FRAME_WIDTH, width);
    capture.set(CV_CAP_PROP_FRAME_HEIGHT, height);

    if(!capture.isOpened()) return -1;

    cv::Mat frame, convertedFrame, diff(height, width, CV_8U);
    capture >> frame;

    if(frame.empty()) return -1;

    std::cout << frame.size() << std::endl;

    cv::VideoWriter outputVideo;

    outputVideo.open("video.mpeg", CV_FOURCC('P','I','M','1'), fps, frame.size(), true);
    
    if (!outputVideo.isOpened())
    {
        std::cout  << "Could not open the output video for write: " << std::endl;
        return -1;
    }

    std::vector<cv::Mat> backgroundImages;
    cv::Mat background;
    const unsigned historySize = 5u;

    initilizeBackground(historySize, background, backgroundImages, capture);

    for(;;)
    {
        capture >> frame;

        cv::cvtColor(frame, convertedFrame, CV_BGR2GRAY);
        if(frame.empty())
            break;

        cv::absdiff(convertedFrame, background, diff);

        cv::threshold(diff, diff, 70, 255, CV_THRESH_BINARY);

        cv::imshow("diff", diff);

        cv::morphologyEx(diff, diff, cv::MORPH_OPEN, cv::Mat::ones(13, 13, CV_8U));

        std::vector<cv::ConnectedComponent> connectedComponents;
        cv::findConnectedComponents(diff, connectedComponents, cv::eightConnected, 255u);

        auto it = std::remove_if(connectedComponents.begin(), connectedComponents.end(),
                                 [minArea](cv::ConnectedComponent const &item)
        {
            return item.getArea() > minArea;
        });

        connectedComponents.erase(it, connectedComponents.end());

        std::for_each(connectedComponents.begin(), connectedComponents.end(),
                      [&frame](cv::ConnectedComponent const &item)
        {
            cv::rectangle(frame, item.getBoundBox(), cv::Scalar(255,0,0), 3);
        });

        time_t  timev;
        std::time(&timev);

        std::string time = currentDateTime();

        cv::putText(frame, time, cv::Point(5, 20),  CV_FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(0,255,255));
        cv::imshow("video", frame);

        sum = cv::sum(diff).val[0];

        if(sum >= alarmThreshhold)
            frameCount = 0;
        else
            frameCount++;


        if(sum > 0)
            std::cout << "diff value " <<std::fixed<<std::setprecision(0)<<std::setw(6)<<sum;
        else
            std::cout << "                 ";


        if(frameCount < saveFrameCount)
        {
            outputVideo << frame;
            std::cout << " recording " << std::endl;

        }else
        {
            std::cout << "          " << std::endl;;
        }

        if(cv::waitKey(1000. / fps - 10) >= 0) break;

    }

    return 0;

}
