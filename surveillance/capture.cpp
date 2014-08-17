#include <ctime>
#include <iomanip>

#include <opencv2/opencv.hpp>


#define HIGH_DEF

const std::string currentDateTime() 
{
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);

    strftime(buf, sizeof(buf), "%Y-%m-%d %X", &tstruct);

    return buf;
}

int main(int, char** )
{
    const int alarmThreshhold = 17000;
    const double fps = 20.0;
#ifdef HIGH_DEF
    const int height = 720;
    const int width = 1280;
#else
    const int height = 480;
    const int width = 640;
#endif

    const int saveFrameCount = 150;

    double sum;
    int frameCount = 0;

    cv::VideoCapture capture(-1);
    capture.set(CV_CAP_PROP_FRAME_WIDTH, width);
    capture.set(CV_CAP_PROP_FRAME_HEIGHT, height);

    if(!capture.isOpened()) return -1;

    cv::Mat frame, convertedFrame, prefFrame = cv::Mat::zeros(height, width, CV_8U), diff(height, width, CV_8U);
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

    for(;;)
    {
        capture >> frame;


        cv::cvtColor(frame, convertedFrame, CV_BGR2GRAY);
        if(frame.empty())
            break;

        cv::absdiff(convertedFrame, prefFrame, diff);

        convertedFrame.copyTo(prefFrame);

        cv::threshold(diff, diff, 70, 255, CV_THRESH_BINARY);

        cv::imshow("diff", diff);

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
