// Cant execute without `pkg-config opencv --cflags --libs`
// Looks like: g++ goshafirsttask.cpp `pkg-config opencv --cflags --libs`

#include "opencv/cv.hpp"
//#include "opencv2/highgui/highui.hpp"
//#include "opencv2/core/core.hpp" //This is the way how we link to opencv?
#include "iostream"

using namespace cv;

int main(int argc, char *argv[])
{
  std::cout<<"Hello World! \n";
  std::cout<<argc<<"\n";
  if (argc!=3)
    {
      std::cout<<"Add input, output file names.\n";
      return -1; //No need for ELSE
    } 
  Mat image = imread("1.jpg");
  putText(image, "This is the one special tree from FFX", Point2f(50,20), FONT_HERSHEY_PLAIN, 1, Scalar(0,0,250,0));
  imshow("First one.", image);
  waitKey(0);
  return 1;
}
