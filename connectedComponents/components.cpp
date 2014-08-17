#include "components.hpp"

namespace cv
{

void findConnectedComponents(const cv::Mat &image, std::vector<cv::ConnectedComponent> connectedComponent,
                             PixelConnectivity pixelConnectivity = fourConnected)

{
    CV_Assert(image.channels() == 1);
}

}
