#include<iostream>
#include<cv.h>
#include<highgui.h>
#include<ml.h>
#include<linux/input.h>
#include<fcntl.h>
//宏定义
//手势数量
const int TNUM = 10;

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
  void Handle_Template();
  //模板载入
  void Handle_Template_Load();
  //模拟鼠标事件
  void Handle_Moving();
  //程序善后工作
  void Handle_Clear();
  //利用svm训练后的数据进行判断
  void Handle_SVM();
  //Attributes
  CvCapture *capture;
  IplImage* src;
  CvSize size;
  CvSeq *handT;
  CvPoint pCenter,cCenter;
  int threshold_Min[3],threshold_Max[3],fd,flag;
  char *Num[10];
  //鼠标事件结构数组
  struct input_event event,event_end;
  //svm识别分类
  CvSVM svm;
  //边界
  CvRect bound;
};
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
  //svm训练
  svm = CvSVM();
  //载入训练文件
  svm.load("thand.xml");
  src = cvQueryFrame( capture );
  size = cvGetSize( src );
  for( int i = 0; i < 3; i++ )
    {
      threshold_Min[ i ] = 0;
      threshold_Max[ i ] = 255;
    }
  memset(&event, 0, sizeof( event ) );
  memset( &event_end, 0, sizeof( event_end ) );
  pCenter = cvPoint( 0, 0 );
  fd = open("/dev/input/event6", O_RDWR );
  //gettimeofday( &event.time, NULL );
  //根据实际
  threshold_Max[0] = 20;
  Init_Windows();
  Handle_Template_Load();
  while(1)
    {
      src = cvQueryFrame( capture );
      Handle_HSV();
      cvShowImage( "video", src );
      Handle_Moving();
      if( cvWaitKey( 10 ) == 27 ) break;
    }
  Handle_Clear();
}
void MyHandle::Init_Windows()
{
  cvNamedWindow( "video", 0 );
  //cvResizeWindow( "video", size.width*1.5, size.height*1.5 );
  cvNamedWindow( "tmp", 0 );
  cvMoveWindow( "tmp", 1000, 750 );
  cvCreateTrackbar( "HL", "tmp", &threshold_Min[0], 255, NULL );
  cvCreateTrackbar( "HH", "tmp", &threshold_Max[0], 255, NULL );
  cvCreateTrackbar( "SL", "tmp", &threshold_Min[1], 255, NULL );
  cvCreateTrackbar( "SH", "tmp", &threshold_Max[1], 255, NULL );
  cvResizeWindow( "tmp", size.width/2, size.height*1.5/2 );
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
  
  //IplConvKernel *tmpCon = cvCreateStructuringElementEx(4,4,0,0,1,0);
  //cvMorphologyEx( dst, dst, 0, tmpCon, CV_MOP_CLOSE, 1 );

  cvShowImage( "tmp", dst );
  Handle_Contours( dst );
  cvReleaseImage( &pHSV );
  cvReleaseImage( &H );
  cvReleaseImage( &S );
  cvReleaseImage( &tS );
  cvReleaseImage( &tH );
  cvReleaseImage( &dst );
}
void MyHandle::Handle_SVM()
{ 
  //绘制
  cvRectangle(src,cvPoint(bound.x-3,bound.y-3),cvPoint(bound.x+3+bound.width,bound.y+bound.height+3),cvScalar(0,0,255,0),6,8,0);
  
}
void MyHandle::Handle_Contours(IplImage *img)
{
  CvMemStorage *storage;
  CvSeq *contours,*tmp;
  double match = 1,t=0,tm=0,j=1;
  //CvContour *contours;
  storage = cvCreateMemStorage( 0 );
  cvFindContours( img, storage, &contours, sizeof(CvContour), CV_RETR_EXTERNAL );
  float imgArea = size.height * size.width;
  for( ;contours != NULL; contours = contours->h_next )
    {
      //decrese the zone by area
      if(fabs( cvContourArea( contours, CV_WHOLE_SEQ ) )/imgArea < 0.02||fabs( cvContourArea( contours, CV_WHOLE_SEQ ) )/imgArea > 0.3  )continue;
      //decrese the zone by position,除去边缘图像
      bound = cvBoundingRect( contours, 0 );
      if(bound.x <= 7 || bound.y <= 7 || bound.x + bound.width >= size.width-6 || bound.y + bound.height >= size.height-6 )continue;      
      //绘制轮廓
      cvDrawContours( src, contours, cvScalar(255,150, 100, 0 ), cvScalar( 255, 0, 0, 0 ), 1, 3, 8, cvPoint( 0, 0 ) );
      //采用svm训练数据进行识别
      Handle_SVM();
      return;
      //下面采用非训练算法，模板匹配
      if( handT == NULL )
	{
	  cout<< "handT == NULL "<<endl;
	  exit(0);
	}
      j = 1;
      for( tmp = handT; tmp; tmp = tmp->h_prev, j++ )
	{
	  tm = cvMatchShapes( tmp, contours, CV_CONTOURS_MATCH_I1, 0 );
	  //get the smallest,系数越小越接近
	  if( tm < match)
	    {
	      match = tm;
	      t = j;
	    }
	}
      //cout<<"==========="<<t<<"----------"<<match<<endl;
      if(match>0.2)continue;
      //get the current position
      cCenter = cvPoint(bound.x, bound.y + bound.height );
      //draw the center point
      cvCircle( src, cCenter,6,cvScalar( 0, 255, 0, 0 ),6, 8, 0);
      //draw the outer frame
      cvRectangle(src,cvPoint(bound.x,bound.y),cvPoint(bound.x+bound.width,bound.y+bound.height),cvScalar(0,0,255,0),6,8,0);
      //绘制文字
      CvFont *font;//外阴影
      cvInitFont(font,CV_FONT_HERSHEY_SIMPLEX,1.0f,1.0f,0,5,8);
      cvPutText(src, Num[((int)t-1)], cvPoint(bound.x, bound.y), font, CV_RGB(255,255,255));		
      //内颜色
      cvInitFont(font,CV_FONT_HERSHEY_SIMPLEX,1.0f,1.0f,0,2,8);
      cvPutText(src, Num[((int)t-1)], cvPoint(bound.x, bound.y), font, CV_RGB(255,0,0));
      //the flag of moving
      //cout<<"++++++++++++++++++"<<t<<endl;
      if( ((int)t - 5 == 0 ) || ( (int)t - 8 == 0 ) )
	{
	  flag = 1;
	  //cout<<"--------flag"<<flag<<endl;
	}
      else
	if ((int)t - 10 == 0 )
	  flag = 2;
	else
	  flag = 0;
      //for(tmp = handT;j!=1;tmp = tmp->h_prev,j--){}
      //cvDrawContours( src, tmp, cvScalar(255,150, 100, 0 ), cvScalar( 255, 0, 0, 0 ), 1, 3, 8, cvPoint( 0, 0 ) );
    }
  cvClearMemStorage( storage );
  //cvClearSeq( contours );
}
void MyHandle::Handle_Template()
{
  
}
void MyHandle::Handle_Template_Load()
{
  CvMemStorage *storage;
  IplImage *tmp;
  CvSeq *t;
  storage = cvCreateMemStorage( 0 );
  //the String in C,when compiling with g++,this throwing warnning
  char* tNum[] = { "1","2","3","4","5","6","7","8","9","10"};
  char* handP[] = {"10.bmp","9.bmp","8.bmp","7.bmp","6.bmp","5.bmp","4.bmp","3.bmp","2.bmp","1.bmp"};
  t = handT = NULL;
  for( int i = 0;i < TNUM; i++)
    {
      tmp = cvLoadImage(handP[i],CV_LOAD_IMAGE_GRAYSCALE);
      if(!tmp)
	{
	  cout<<"NOT Found File: "<< handP[i] << endl;
	  continue;
	}
      Num[i] = tNum[i];
      cvFindContours( tmp, storage, &t, sizeof(CvContour), CV_RETR_EXTERNAL );
      if( t )
	{
	  cout<< "Load The Template " << handP[ i ] << " Success "<< endl;
	  if( handT == NULL )
	    {
	      cout<< "Load First Template " << endl;
	      handT = t;
	    }
	  else
	    {
	      t->h_prev = handT;
	      handT -> h_next = t;
	      handT = handT -> h_next;
	    }
	}
    }
  cvClearMemStorage( storage );
  cvReleaseImage( &tmp );
}
//my operation system is centos 6.4 
void MyHandle::Handle_Moving()
{
  if( !fd )
    {
      cout<< "Error To Open Mouse" <<endl;
      return;
    }
  if( !flag )
    {
      cout<< "Not The Moving Flag "<< endl;
      pCenter = cCenter;
      return;
    }
  //cout<< "flag-----------"<< flag <<endl;
  //if( abs(cCenter.x - pCenter.x)<
  gettimeofday( &event.time, NULL );
  gettimeofday( &event_end.time, NULL );
  //Click The Mouse Button
  event.type = EV_KEY;
  event.value = 0;
  event.code = BTN_LEFT;
  if( flag == 2 )
    {
      //cout<<"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"<<endl;
      //event.type = EV_KEY;
      //if( event.code != BTN_LEFT )
      //{
      //  sleep(1);
      //}
      event.value = 1;
      //event.code = BTN_LEFT;
    }
  write( fd, &event, sizeof( event ) );
  write( fd, &event_end, sizeof( event_end ) );
  //Moving The Mouse
  event.type = EV_REL;
  double offset = abs(cCenter.x - pCenter.x); 
  if( offset > 2 )
    {
      event.code = REL_X;
      event.value = 0 - ( offset > 4 ? 5.5 : 1 )*(cCenter.x - pCenter.x);
      event_end.type = EV_SYN;
      event_end.code = SYN_REPORT;
      event_end.value = 0;
      write( fd, &event, sizeof( event ) );
      write( fd, &event_end, sizeof( event_end ) );
      flag = 10;
    }
  offset = abs(cCenter.y - pCenter.y);
  if( offset > 2 )
    {
      event.code = REL_Y;
      event.value = ( offset > 3 ? 4 : 1 )*(cCenter.y - pCenter.y );
      write( fd, &event, sizeof( event ) );
      write( fd, &event_end, sizeof( event_end ) );
      flag = 10;
    }
  if( flag == 10 )
    pCenter = cCenter;
  flag = 0;
}
void MyHandle::Handle_Clear()
{
  cvDestroyWindow("video");
  cvReleaseCapture( &capture );
  cvReleaseImage( &src );
  //close the file
  close( fd );
}
int main(int argc,char** argv)
{
  MyHandle handle(0);
  return 0;
}
