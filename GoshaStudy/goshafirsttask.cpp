// Cant execute without `pkg-config opencv --cflags --libs`
// Looks like: g++ goshafirsttask.cpp `pkg-config opencv --cflags --libs`

#include "opencv/cv.hpp"
//#include "opencv2/highgui/highui.hpp"
//#include "opencv2/core/core.hpp" //This is the way how we link to opencv?
#include "iostream"

using namespace cv;

int main(int argc, char *argv[])
{
  if (argc!=3)
    {
      std::cout<<"Add input, output file names.\n";
      return -1; //No need for ELSE
    } 
  Mat image = imread(argv[1]);
  if (image.type()!=CV_8UC3)
    {
      std::cout<<"Not an image.";
      return -1;
    }
  putText(image, "This is the one special tree from FFX", Point2f(50,20), FONT_HERSHEY_PLAIN, 1, Scalar(0,0,250,0));
  rectangle(image, Point2f(450,10),Point2f(500,20), Scalar(25,250,25,50));
  imshow("First one.", image);
  waitKey(0);
  imwrite(argv[2], image);
  return 1;
}
