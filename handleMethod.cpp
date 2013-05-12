#include "handleClass.h"
#include<iostream>
#include<cv.h>
#include<highgui.h>

using namespace std;
using namespace cv;

MyHandle::MyHandle(int camIndex)
{
  if(Init_Cam(camIndex))
    cout<<"Could Not Found Camera"<<endl;
  else
    Handle_Capture();
}
int MyHandle::Init_Cam(int camIndex)
{
  return (capture = cvCaptureFromCAM( camIndex ))==NULL ? 1 : 0;
}
void MyHandle::Handle_Capture()
{
  src = cvQueryFrame( capture );
  size = cvGetSize( src );
  for( int i = 0; i < 3; i++ )
    {
      threshold_Min[ i ] = 0;
      threshold_Max[ i ] = 255;
    }
  Init_Windows();
  while(1)
    {
      src = cvQueryFrame( capture );
      Handle_HSV();
      cvShowImage( "video", src );
      if( cvWaitKey( 10 ) == 27 ) break;
    }
}
void MyHandle::Init_Windows()
{
  cvNamedWindow( "video", 0 );

  cvNamedWindow( "tmp", 0 );
  cvMoveWindow( "tmp", 1000, 750 );
  cvCreateTrackbar( "HL", "tmp", &threshold_Min[0], 255, NULL );
  cvCreateTrackbar( "HH", "tmp", &threshold_Max[0], 255, NULL );
  cvCreateTrackbar( "SL", "tmp", &threshold_Min[1], 255, NULL );
  cvCreateTrackbar( "SH", "tmp", &threshold_Max[1], 255, NULL );
  cvResizeWindow( "tmp", size.width, size.height*1.5 );
}
void MyHandle::Handle_HSV()
{
  IplImage *pHSV,*H,*S,*tH,*tS,*dst;
  pHSV = H = S = tH = tS = dst = NULL;
  pHSV = cvCreateImage( size, IPL_DEPTH_8U, 3 );
  H = cvCreateImage( size, IPL_DEPTH_8U, 1 );
  S = cvCreateImage( size, IPL_DEPTH_8U, 1 );
  tH = cvCreateImage( size, IPL_DEPTH_8U, 1 );
  tS = cvCreateImage( size, IPL_DEPTH_8U, 1 );
  dst = cvCreateImage( size, IPL_DEPTH_8U, 1 );
  cvCvtColor( src, pHSV, CV_BGR2HSV );
  cvSplit( pHSV, H, S, 0, 0 );
  cvInRangeS( H, cvScalar( threshold_Min[0], 0, 0, 0 ), cvScalar( threshold_Max[0], 0, 0, 0 ), tH );
  cvInRangeS( S, cvScalar( threshold_Min[1], 0, 0, 0 ), cvScalar( threshold_Max[1], 0, 0, 0 ), tS );
  cvAnd( tH, tS, dst, 0 );
  cvErode( dst, dst, 0, 2 );
  cvDilate( dst, dst, 0, 1 );
  cvShowImage( "tmp", dst );
  Handle_Contours( dst );
  cvReleaseImage( &pHSV );
  cvReleaseImage( &H );
  cvReleaseImage( &S );
  cvReleaseImage( &tS );
  cvReleaseImage( &tH );
  cvReleaseImage( &dst );
}
void MyHandle::Handle_Contours(IplImage *img)
{
  CvMemStorage *storage;
  CvSeq *contours;
  //CvContour *contours;
  storage = cvCreateMemStorage( 0 );
  cvFindContours( img, storage, &contours, sizeof(CvContour), CV_RETR_EXTERNAL );
  float imgArea = size.height * size.width;
  CvRect bound;
  for( ;contours != NULL; contours = contours->h_next )
    {
      if(fabs( cvContourArea( contours, CV_WHOLE_SEQ ) )/imgArea < 0.02 )continue;
      //cvDrawContours( img, contours, cvScalar(255,150, 100, 0 ), cvScalar( 255, 0, 0, 0 ), 1, 10, 8, cvPoint( 0, 0 ) );
      bound = cvBoundingRect( contours, 0 );
      cvRectangle(src,cvPoint(bound.x,bound.y),cvPoint(bound.x+bound.width,bound.y+bound.height),cvScalar(0,0,255,0),2,8,0);
    }
  cvClearMemStorage( storage );
  //cvClearSeq( contours );
}
void MyHandle::Handle_Clear()
{
  cvDestroyWindow("video");
  cvReleaseCapture( &capture );
  cvReleaseImage( &src );
}
