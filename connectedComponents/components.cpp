#include "components.hpp"

namespace cv
{


std::vector<cv::Point2i> priorNeighbors(unsigned x, unsigned y, const cv::Mat &image, PixelConnectivity pixelConnectivity)
{
    std::vector<cv::Point2i> neighbors;

    cv::Rect imageRect(cv::Point2i(0,0), image.size());

    if(pixelConnectivity == cv::eightConnected)
    {
        cv::Point2i point_1(y - 1, x - 1);
        if(imageRect.contains(point_1))
            if(image.at<short>(point_1))
                neighbors.push_back(point_1);


        cv::Point2i point_2(y+1, x-1);
        if(imageRect.contains(point_2))
            if(image.at<short>(point_2))
                neighbors.push_back(point_2);

    }

    cv::Point2i point_1(y, x - 1);
    if(imageRect.contains(point_1))
        if(image.at<short>(point_1))
            neighbors.push_back(point_1);

    cv::Point2i point_2(y - 1, x);
    if(imageRect.contains(point_2))
        if(image.at<short>(point_2))
            neighbors.push_back(point_2);

    return neighbors;
}

short getLabel(const std::vector<Point2i> &neighbors, const cv::Mat &image, UnionSearch &unionSearch)
{

    CV_Assert(image.type() == CV_16S);

    short actMin {std::numeric_limits<short>::max()};

    for(const auto &neighbor : neighbors)
        actMin = std::min(actMin, image.at<short>(neighbor));

    CV_Assert(actMin != std::numeric_limits<short>::max());

    for(const auto &neighbor : neighbors)
        if(actMin != image.at<short>(neighbor))
            unionSearch.merge(actMin, image.at<short>(neighbor));

    return actMin;
}

void findConnectedComponents(const cv::Mat &image, std::vector<cv::ConnectedComponent> &connectedComponents,
                             PixelConnectivity pixelConnectivity = cv::fourConnected, uchar foregroundValue = 255u)

{
    CV_Assert(image.channels() == 1);

    cv::Mat inverted; image.convertTo(inverted, CV_16S, -1.);

    cv::UnionSearch unionSearch;

    short unlabaled {foregroundValue}, currentLabel {1};

    for(unsigned y = 0u; y < inverted.rows; ++y)
    {
        for(unsigned x = 0u; x < inverted.cols; ++x)
        {
            if(inverted.at<short>(y,x) == -unlabaled)
            {
                std::vector<Point2i> neighbors = priorNeighbors(y, x, inverted, pixelConnectivity);

                if( !neighbors.empty() )
                {
                    short label = getLabel(neighbors, inverted, unionSearch);
                    inverted.at<short>(y,x) = label;
                }else
                {
                    inverted.at<short>(y,x) = currentLabel;
                    unionSearch.add(currentLabel++);
                }
            }
        }
    }

    std::map<unsigned, unsigned> adapter;
    unsigned count {};

    for(unsigned y = 0u; y < inverted.rows; ++y)
    {
        for(unsigned x = 0u; x < inverted.cols; ++x)
        {
            CV_Assert(inverted.at<short>(y,x) >= 0);
            if(inverted.at<short>(y,x) != 0)
            {
                cv::Point2i point(x,y);
                unsigned parent = unionSearch.findParent(inverted.at<short>(point));

                if(adapter.find(parent) == adapter.end())
                {
                    connectedComponents.push_back(cv::ConnectedComponent());
                    adapter[parent] = count++;

                }

                connectedComponents[adapter[parent]].addPoint(point);

            }
        }
    }

    for(auto &component : connectedComponents)
    {
        for(const auto &point : component.getPoints())
        {
            if(cv::isBound(point, inverted, pixelConnectivity))
                component.addBoundPoint(point);
        }
    }
}


UnionSearch::UnionSearch()
{
    storage.push_back(0);
}

void UnionSearch::add(short element)
{
    CV_Assert(storage.size() == element);
    storage.push_back(0);
}

unsigned UnionSearch::findParent(short element)
{
    CV_Assert(element >= 1);
    CV_Assert(element < storage.size());

    unsigned parent = static_cast<unsigned>(element);

    while(storage[parent])
    {
        parent = static_cast<unsigned>(storage[parent]);
    }

    return parent;
}

void UnionSearch::merge(short element_1, short element_2)
{
    auto parent_1 = findParent(element_1);
    auto parent_2 = findParent(element_2);

    CV_Assert(parent_1 != 0 && parent_2 != 0);

    if(parent_1 == parent_2)
        return;

    storage[parent_1] = parent_2;
}

void ConnectedComponent::addPoint(const Point2i &point)
{
    points.push_back(point);
}

void ConnectedComponent::draw(Mat &image, cv::RNG &rng) const
{
    int iColor_rng = (unsigned) rng;
    Scalar iColor(iColor_rng&255, (iColor_rng>>8)&255, (iColor_rng>>16)&255);

    for (const auto &point : points)
        cv::circle(image, point, 1, iColor);


    int bColor_rng = (unsigned) rng;
    Scalar bColor(bColor_rng&255, (bColor_rng>>8)&255, (bColor_rng>>16)&255);

    for (const auto &bPoint : boundPoints)
        cv::circle(image, bPoint, 1, bColor);

    cv::Point2d center = calcCenter();
    cv::circle(image, center, 2, bColor, 6);

    std::ostringstream oss;  oss << roundness_1();

    //oss << std::setw(2) << std::setprecision(2);

    cv::putText(image, oss.str(), center, cv::FONT_HERSHEY_PLAIN, 1, bColor);
}

const std::vector<Point2i> &ConnectedComponent::getPoints() const
{
    return points;
}

void ConnectedComponent::addBoundPoint(const Point2i &point)
{
    boundPoints.push_back(point);
}

unsigned ConnectedComponent::calcPerimeter() const
{
    return boundPoints.size();
}

unsigned ConnectedComponent::calcArea() const
{
    return points.size();
}

Point2d ConnectedComponent::calcCenter() const
{
    cv::Point2d center(0.,0.);

    for(const auto & point: points)
        center +=  cv::Point2d(point.x, point.y);

    center.x = center.x / calcArea();
    center.y = center.y / calcArea();

    return center;
}

double ConnectedComponent::roundness_1() const
{
    return static_cast<double>( calcPerimeter() * calcPerimeter()) / calcArea();
}

double ConnectedComponent::roundness_2() const
{

}

double ConnectedComponent::centralSecondMomentRow() const
{

}

double ConnectedComponent::centralSecondMomentCols() const
{

}

double ConnectedComponent::mixedCentralMoment() const
{

}

void drawConnectedComponents(Mat &image, const std::vector<ConnectedComponent> &connectedComponents)
{
    CV_Assert(image.type() == CV_8UC3);

    cv::RNG rng( 0xFFFFFFFF );

    for(const auto &connectedComponent : connectedComponents)
        connectedComponent.draw(image, rng);
}

bool isBound(const Point2i &point, const Mat &image, PixelConnectivity pixelConnectivity)
{
    cv::Rect imageRect(cv::Point2i(0,0), image.size());

    std::vector<cv::Point2i> dPoints {cv::Point2i(-1,0), cv::Point2i(1,0), cv::Point2i(0,-1), cv::Point2i(0,1)};

    if(pixelConnectivity == cv::eightConnected)
    {
        dPoints.push_back(cv::Point2i(-1,-1));
        dPoints.push_back(cv::Point2i(-1, 1));
        dPoints.push_back(cv::Point2i( 1,-1));
        dPoints.push_back(cv::Point2i( 1, 1));
    }

    for( const auto &dPoint : dPoints)
        if(imageRect.contains(point + dPoint))
            if(image.at<short>(point + dPoint) == 0)
                return true;

    return false;
}

}
