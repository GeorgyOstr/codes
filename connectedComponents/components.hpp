#include <opencv2/opencv.hpp>

namespace cv
{
    class ConnectedComponent
    {
    public:

    private:
        std::vector<cv::Point2i> points;
    };

    enum PixelConnectivity { fourConnected, eightConnected };

    void findConnectedComponents(const cv::Mat &image, std::vector<cv::ConnectedComponent> connectedComponent,
                                 PixelConnectivity pixelConnectivity);
}
