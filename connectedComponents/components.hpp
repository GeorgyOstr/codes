#include <opencv2/opencv.hpp>

#include <iomanip>

namespace cv
{
    class ConnectedComponent
    {
    public:
        void addPoint(const cv::Point2i &point);
        void draw(Mat &image, RNG &rng) const;
        const std::vector<cv::Point2i> &getPoints() const;
        void addBoundPoint(const cv::Point2i &point);

        unsigned calcPerimeter() const;
        unsigned calcArea() const;
        Point2d calcCenter() const;

        double roundness_1() const;
        double roundness_2() const;

        std::tuple<double, double> centralSecondMomentRowNCols() const;
        double mixedCentralMoment() const;

        cv::Rect getBoundBox() const;

    private:
        std::vector<cv::Point2i> points;
        std::vector<cv::Point2i> boundPoints;
    };

    enum PixelConnectivity { fourConnected, eightConnected };

    void findConnectedComponents(const cv::Mat &image, std::vector<cv::ConnectedComponent> &connectedComponents,
                                 PixelConnectivity pixelConnectivity, uchar foregraoundValue);

    void drawConnectedComponents(cv::Mat &image, const std::vector<ConnectedComponent> &connectedComponents);

    bool isBound(const cv::Point2i &point, const cv::Mat &image, PixelConnectivity pixelConnectivity);

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
