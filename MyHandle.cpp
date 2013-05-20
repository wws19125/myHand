#include<iostream>
#include<cv.h>
#include<highgui.h>
#include<ml.h>
#include<linux/input.h>
#include<fcntl.h>
#include<sys/types.h>
#include<sys/stat.h>
#include<vector>
#include<pthread.h>
#include<stdio.h>
//宏定义
#define FIFO_NAME "/tmp/opencv_fifo"
//键盘事件
#define Key_Event "/dev/input/event3"
//鼠标事件
#define Mouse_Event "/dev/input/event12"
//手势模板数量
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
  //创建管道，为进程通信准备
  void Handle_Pipe();
  //获取训练数据
  void Handle_Collection();
  //训练
  void Handle_Train();
  //预测
  void Handle_Predict();
  //处理手势
  void Handle_Guesture(int Guesture);
  //模拟按键
  void Handle_Simulate_key(int file, int keytype, unsigned int keycode, int keyvalue);
  //ctrl
  void Handle_Simulate_Ctrl_key(int file,int keytype,unsigned int keycode);
  //计算偏移
  void Handle_Offset();
  //模拟鼠标
  void Handle_Simulate_Mouse();


  //Attributes
  CvCapture *capture;
  IplImage* src;
  //目标
  IplImage* Roi;
  IplImage *trainImg;
  CvSize size;
  CvSeq *handT;
  //记录的移动点
  CvPoint pCenter,cCenter;
  //偏移
  int offset[2];
  int threshold_Min[3],threshold_Max[3],fd,flag,k_fd;
  char *Num[10];
  //边界
  CvRect bound;
  //鼠标事件结构数组
  struct input_event event,event_end;
  //svm识别分类
  CvSVM svm;
  CvSVMParams param;
  CvTermCriteria criteria;
  //训练用
  vector<float> res;
  HOGDescriptor *hog;
  vector<int> cat_res;
  vector<float> t_hog;
  CvMat *data_mat,*res_mat;
  //训练分类，训练数量
  int pClass,pNum;
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
  pClass = 0;
  pNum = 0;
  trainImg = cvCreateImage(cvSize(64,64),8,3);
  hog = new HOGDescriptor(cvSize(64,64),cvSize(16,16),cvSize(8,8),cvSize(8,8),9);
  //svm训练
  svm = CvSVM();
  criteria = cvTermCriteria( CV_TERMCRIT_EPS, 1000, FLT_EPSILON );      
  param = CvSVMParams( CvSVM::C_SVC, CvSVM::RBF, 10.0, 0.09, 1.0, 10.0, 0.5, 1.0, NULL, criteria );

  //载入训练文件
  //svm.load("thand.xml");
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
  fd = open(Mouse_Event, O_RDWR );
  k_fd = open(Key_Event,O_RDWR);
  //gettimeofday( &event.time, NULL );
  //根据实际,此处判断
  threshold_Max[0] = 20;
  Init_Windows();
  //初始化记录点
  cCenter = cvPoint(0,0);
  pCenter = cvPoint(0,0);
  //加载模板
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
  cvNamedWindow("block", 0 );
  cvMoveWindow( "block", 0, 750 );
  cvCreateTrackbar( "pClass", "block", &pClass, 8, NULL );
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
  //获取感兴趣的部分
  Roi = cvCreateImageHeader(cvSize(bound.width+20,bound.height+20),src->depth,src->nChannels);
  Roi->origin = src->origin;
  Roi->widthStep = src->widthStep;
  Roi->imageData = src->imageData + (bound.y - 10)*src->widthStep + ( bound.x -10 )*src->nChannels;
  cvShowImage( "block", Roi );
  // s 按键
  if(cvWaitKey(10) == 115)
    {
      //进程通信，发送到训练进程去
      //msg.img = Roi;
      //Handle_Pipe();
      Handle_Collection();
    }
  else
    //检测
    Handle_Predict();
  //绘制
  cvRectangle(src,cvPoint(bound.x-3,bound.y-3),cvPoint(bound.x+3+bound.width,bound.y+bound.height+3),cvScalar(0,0,255,0),6,8,0);
 }
void MyHandle::Handle_Collection()
{
  cvShowImage( "block", Roi );
  if(cvWaitKey(0)==111)
    {
      cvResize( Roi, trainImg);
      hog->compute( trainImg,res,Size(1,1),Size(0,0));
      for(vector<float>::iterator item = res.begin();item != res.end(); item++)
	{
	  t_hog.push_back(*item);
	}
      cat_res.push_back(pClass);
      cvReleaseImage( &Roi );
      cout<<res.size()<<"  HOG"<<endl;
      cout<<"the num is "<< ++pNum <<endl;
      cout<<"the class "<<pClass<<endl;
      Handle_Train();
    }
}
void MyHandle::Handle_Train()
{
  if(cat_res.size()==1)return;
  data_mat = cvCreateMat(pNum,1764,CV_32FC1);
  res_mat = cvCreateMat(pNum,1,CV_32FC1);
  //cvInitMatHeader (&res_mat, all_imnum, 1, CV_32SC1, res);
  cvSetZero( res_mat );
  cvSetZero( data_mat );
  int t,i;
  t=0;
  i=0;
  for( vector<float>::iterator it  = t_hog.begin(); it != t_hog.end(); it++ )
    {
      cvmSet( data_mat, t, i, *it );
      if( ++i == 1764 )
	{
	  t++;
	  i=0;
	}
    }
  for( i = 0; i < cat_res.size(); i++ )
    {
      cout<<cat_res[i]<<endl;
      cvmSet( res_mat, i, 0, cat_res[i] ); 
    }
  cout<<"start to trainning"<<endl;
  svm.train(data_mat,res_mat,NULL,NULL,param);
  svm.save("thand.xml");
}
void MyHandle::Handle_Predict()
{
  svm.load("thand.xml");
  cvZero(trainImg);
  cvResize(Roi,trainImg);
  vector<float> pres;
  hog->compute(trainImg, pres,Size(1,1), Size(0,0)); 
  CvMat* SVMtrainMat=cvCreateMat(1,pres.size(),CV_32FC1); 
  int n=0;    
  for(vector<float>::iterator iter=pres.begin();iter!=pres.end();iter++)    
    {    
      cvmSet(SVMtrainMat,0,n,*iter);    
      n++;    
    }
  //预测结果
  Handle_Guesture(svm.predict(SVMtrainMat));
  return;
  //绘制文字
  CvFont *font;//外阴影
  char c = '0'+n;
  cvInitFont(font,CV_FONT_HERSHEY_SIMPLEX,1.0f,1.0f,0,5,8);
  cvPutText(src, &c, cvPoint(bound.x, bound.y), font, CV_RGB(255,255,255));		
  //内颜色
  cvInitFont(font,CV_FONT_HERSHEY_SIMPLEX,1.0f,1.0f,0,2,8);
  cvPutText(src, &c, cvPoint(bound.x, bound.y), font, CV_RGB(255,0,0));
}
//处理手势,需要使用系统文件，此处不可以移植到其他平台
void MyHandle::Handle_Guesture(int Guesture)
{
  Handle_Offset();
  /*
    Guesture
    0 握拳        移动
    1 一根手指     鼠标
    2 两根手指     -1
    3 三根手指     -1
    4 四根手指     -1
    5 五根手指     backspace/space
    6 弯曲手指     点击
    7 合拢手指     放大
    8 分开手指     缩小
  */ 
  cout<<"the Guesture is "<<Guesture<<endl;
  switch(Guesture)
    {
    case 0:
      
      return;
    case 1:
      if(fd == -1 )
	{
	  cout<<"please check you mouse event"<<endl;
	  return;
	}
      Handle_Simulate_Mouse();      
      //更新记录点
      pCenter = cCenter;
      return;
    case 2:
      return;
    case 3:
      return;
    case 4:
      return;
    case 5:
      //打开失败
      if(k_fd==-1)
	{
	  cout<<"please check you device"<<endl;
	  return;
	}
      if(offset[0]>110||offset[1]>110)
	{
	  cout<<"================================space"<<endl;
	  Handle_Simulate_key( k_fd, EV_KEY,KEY_SPACE,1);
	}
      else
	if(offset[0]<-110||offset[1]<-110)
	  {
	    cout<<"==============================backspace"<<endl;
	    Handle_Simulate_key( k_fd, EV_KEY,KEY_BACKSPACE,1);
	  }
      return;
    case 6:
      //限制点击次数
      //if(++cNum[6]>=3)
      //{
      //  cout<<"==========================limit click"<<endl;
      //  return;
      //}
      if(fd == -1 )
	{
	  cout <<"please check you device"<<endl;
	  return;
	}
      Handle_Simulate_key( fd,EV_KEY, BTN_LEFT, 1 );
      //延时用于处理双击
      usleep(50000);
      return;
    case 7:
      
      return;
    case 8:
      return;
    }
  //更新记录点
  pCenter = cCenter;
}
//模拟按键
void MyHandle::Handle_Simulate_key(int file, int keytype,unsigned int keycode, int keyvalue)
{
  event.type = keytype;
  event.code = keycode;
  event.value = keyvalue;
  gettimeofday(&event.time,0);
  write(file,&event,sizeof(event));
  memset(&event, 0, sizeof(event));
  gettimeofday(&event.time,0);
  event.type = keytype;
  event.code = keycode;
  event.value = 0;
  write(file,&event,sizeof(event));
  event.type = EV_SYN;
  event.code = SYN_REPORT;
  event.value = 0;
  write(file,&event,sizeof(event));
}
void MyHandle::Handle_Simulate_Ctrl_key(int file,int keytype,unsigned int keycode)
{
}
//模拟鼠标
void MyHandle::Handle_Simulate_Mouse()
{
  //cout<<"Moving the mouse"<<endl;
  //处理抖动
  if(offset[0]==offset[1]==1)return;
  //移动鼠标
  Handle_Simulate_key( fd, EV_REL, REL_X, abs(offset[0]>6 ? -7 : 1-abs(offset[0]) )*offset[0] );
  Handle_Simulate_key( fd, EV_REL, REL_Y, abs(offset[1]>5 ? 5 : 1 )*offset[1] );
  
}
//进程通信(暂时废弃)
void MyHandle::Handle_Pipe()
{
 
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
      //cvDrawContours( src, contours, cvScalar(255,150, 100, 0 ), cvScalar( 255, 0, 0, 0 ), 1, 3, 8, cvPoint( 0, 0 ) );
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
      //cout<< "Not The Moving Flag "<< endl;
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
void MyHandle::Handle_Offset()
{  
  //get the current position
  cCenter = cvPoint(bound.x, bound.y + bound.height );
  // x
  offset[0] = cCenter.x - pCenter.x;
  // y
  offset[1] = cCenter.y - pCenter.y;
}
void MyHandle::Handle_Clear()
{
  destroyAllWindows();
  //cvDestroyWindow("video");
  cvReleaseCapture( &capture );
  cvReleaseImage( &src );
  cvReleaseImage( &Roi );
  cvReleaseImage( &trainImg );
  //close the file
  close( fd );
  close( k_fd );
}
int main(int argc,char** argv)
{
  MyHandle handle(0);
  return 0;
}
