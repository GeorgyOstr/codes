#include <opencv2/opencv.hpp>

#include <components.hpp>
#include <surveillance.hpp>

int main(int, char** )
{
    const double fps = 20.0;
    const int saveFrameCount = 150;
    const unsigned minArea = 15;
    const cv::Size kernelSize(5,5);
    const double minRoundness = 2.;
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

        cv::imshow("old_diff", diff);

        cv::morphologyEx(diff, diff, cv::MORPH_CLOSE, cv::Mat::ones(kernelSize, CV_8U));

        cv::imshow("new_diff", diff);

        std::vector<cv::ConnectedComponent> connectedComponents;
        cv::findConnectedComponents(diff, connectedComponents, cv::eightConnected, 255u);

        auto it = std::remove_if(connectedComponents.begin(), connectedComponents.end(),
                                 [minArea, minRoundness](cv::ConnectedComponent const &item)
        {
            double roundness_2 = item.roundness_2();
            if(std::isnan(roundness_2) || std::isfinite(roundness_2))
                return true;

            return item.getArea() <= minArea && roundness_2 < minRoundness;
        });

        connectedComponents.erase(it, connectedComponents.end());

        if(!connectedComponents.empty())
        {
            std::for_each(connectedComponents.begin(), connectedComponents.end(),
                          [&frame](cv::ConnectedComponent const &item)
            {
                cv::rectangle(frame, item.getBoundBox(), cv::Scalar(0,255,0), 5);
            });

            frameCount = 0;
        }
        else
        {
            frameCount++;
        }

        cv::drawConnectedComponents(frame, connectedComponents);

        time_t  timev;
        std::time(&timev);

        std::string time = currentDateTime();

        cv::putText(frame, time, cv::Point(5, 20),  CV_FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(0,255,255));
        if(frameCount < saveFrameCount)
        {
            outputVideo << frame;
            cv::circle(frame, cv::Point(15, 40), 5, cv::Scalar(0,0,255),10);
        }

        cv::imshow("video", frame);

        //if(cv::waitKey(1000. / fps - 10) >= 0) break;

        if(cv::waitKey(1500) >= 0) break;
    }

    return 0;

}
