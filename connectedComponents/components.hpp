#include <opencv2/opencv.hpp>

namespace cv
{
    class ConnectedComponent
    {
    public:
        void addPoint(const cv::Point2i &point);
        void draw(Mat &image, RNG &rng) const;

    private:
        std::vector<cv::Point2i> points;
    };

    enum PixelConnectivity { fourConnected, eightConnected };

    void findConnectedComponents(const cv::Mat &image, std::vector<cv::ConnectedComponent> &connectedComponents,
                                 PixelConnectivity pixelConnectivity, uchar foregraoundValue);

    void drawConnectedComponents(cv::Mat &image, const std::vector<cv::ConnectedComponent> &connectedComponents);

    class UnionSearch
    {
    public:
        UnionSearch();
        void add(short element);
        unsigned findParent(short element);
        void merge(short element_1, short element_2);
    private:
        std::vector<short> storage;

    };
}
