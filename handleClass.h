
#ifndef _handleClass_H_
#define _handleClass_H_

#include <cv.h>
#include <highgui.h>

using namespace std;
using namespace cv;

class MyHandle
{
public:
  MyHandle(int camIndex);
private:
  //Methods
  int Init_Cam(int camIndex);
  void Handle_Capture();
  void Init_Windows();
  void Handle_HSV();
  void Handle_HSV_Threshold();
  void Handle_Contours( IplImage *img );
  void Handle_Clear();
  //Attributes
  CvCapture *capture;
  IplImage* src;
  CvSize size;
  int threshold_Min[3],threshold_Max[3];
};

#endif
