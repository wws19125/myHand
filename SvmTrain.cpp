#include<iostream>
//#include<stdio.h>
#include<cv.h>
#include<highgui.h>
#include<ml.h>
#include<vector>
#include<fcntl.h>
using namespace std;
using namespace cv;

class Message
{
public:
  int id;
  IplImage img;
};

class SvmTrain
{
public:
  SvmTrain(int camIndex);
  SvmTrain();
private:
  //attributes
  //捕获摄像头
  CvCapture *capture;   
  //源图像
  IplImage *src,*tmp,*trainImg;
  vector<float> res;
  int tt[10];
  HOGDescriptor *hog;
  //分类结果
  static vector<int> cat_res;
  //HOG暂存
  static vector<float> t_hog;
  //HOG存放
  CvMat *data_mat,*res_mat;
  //临时分类   图片数量
  int pClass;
  static int pNum;
  //是否绘制
  static bool drawingBox;
  //要绘制的矩形框
  static CvRect box;
  //训练用
  CvSVM svm;
  CvSVMParams param;
  CvTermCriteria criteria;
  //methods
  /*检测摄像头
    @param camIndex  摄像头索引
    @res   捕获结果         
   */
  bool initCam(int camIndex);
  //初始化窗口
  void initWin();
  //从文件中载入
  void loadSampleFromFile();
  
  //从摄像头中获取训练样本
  void getSampleFromCAM();
  //鼠标截取
  void getZoneByMouse();
  //鼠标回调事件
  static void mouseCallback( int event, int x, int y, int flags, void* param );
  //绘制矩形框
  void getRoi();
  //采取数据
  void collectData(IplImage* ROi);
  //开始训练
  void process_Train();
  //初始化
  void init();
  //清理垃圾
  void clearMem();
  
};
int SvmTrain::pNum = 0;
vector<int> SvmTrain::cat_res;
vector<float> SvmTrain::t_hog;
//Mat SvmTrain::data_mat;
//Mat SvmTrain::res_mat;
CvRect SvmTrain::box = cvRect( -1, -1, 0, 0 );
bool SvmTrain::drawingBox = false;
//进程间通信
SvmTrain::SvmTrain()
{
  if(access("/tmp/opencv_fifo",F_OK)==-1)
    {
      cout<<"no pipe"<<endl;
      exit(0);
    }
  int fd;
  IplImage img;
  cout<<img.nSize<<endl;
  init();
  cout<<"prepare to read"<<endl;
  while(1)
    {
      //读取管道
      if((fd=open("/tmp/opencv_fifo",O_RDONLY)) != -1)
	{
	  cout<<read(fd,&img,200)<<endl;
      //cvShowImage( "block",&img);	  
	  cout<<img.nSize<<endl;
	  cvShowImage( "block",&img);
	}
      cout<<endl;
    }
}
SvmTrain::SvmTrain(int camIndex)
{
  if(initCam(camIndex))
    return;
  init();
  getSampleFromCAM();
}
void SvmTrain::init()
{
  //data_mat = Mat::zeros( 1000,
  pClass=0;
  //pNum=0;
  trainImg = cvCreateImage(cvSize(64,64),8,3);
  hog = new HOGDescriptor(cvSize(64,64),cvSize(16,16),cvSize(8,8),cvSize(8,8),9);
  svm = CvSVM();//新建一个SVM
  criteria = cvTermCriteria( CV_TERMCRIT_EPS, 1000, FLT_EPSILON );      
  param = CvSVMParams( CvSVM::C_SVC, CvSVM::RBF, 10.0, 0.09, 1.0, 10.0, 0.5, 1.0, NULL, criteria );
  initWin();  
}
bool SvmTrain::initCam(int camIndex)
{
  return (capture = cvCaptureFromCAM( camIndex )) == NULL ? true : false;
}
void SvmTrain::initWin()
{
  //源窗口
  cvNamedWindow( "src", 0 );
  //鼠标截图
  cvNamedWindow( "block", 0 );
  cvCreateTrackbar( "pClass", "block", &pClass, 8, NULL );
  cvMoveWindow("block",350,0);
  //结果
  //cvNamedWindow( "res", 0 );
}
void SvmTrain::loadSampleFromFile()
{
  cout<<box.x<<endl;
}
void SvmTrain::getSampleFromCAM()
{
  
  src = cvQueryFrame( capture );
  while(true)
    {
      cvShowImage( "src", src);
      if( cvWaitKey( 10 ) == 27 ) break;
      src = cvQueryFrame( capture );
      getZoneByMouse();
    }
}
void SvmTrain::getZoneByMouse()
{
  //IplImage *img = cvCreateImage( cvSize(200,200), IPL_DEPTH_8U, 3 );
  //Zero( img );
  //IplImage *tmp = cvCloneImage( src );
  //cvNamedWindow( "box" );
  cvSetMouseCallback( "src", mouseCallback, (void*)this );
  //while(true)
  //{
  //  cvCopyImage( src, tmp );
      //if( drawingBox ) drowBox( tmp, box );
      //cvShowImage( "box", tmp );
  //  if( cvWaitKey( 15 ) == 27 ) break;
  //}
  //cvReleaseImage( &tmp );  
}
void SvmTrain::mouseCallback( int event, int x, int y, int flags, void* param )
{  
  SvmTrain st = *(SvmTrain*)param;
  switch( event )
    {
    case CV_EVENT_MOUSEMOVE:
      if( drawingBox )
	{
	  box.width = x - box.x;
	  box.height = y - box.y;
	}
      break;
    case CV_EVENT_LBUTTONDOWN:
      drawingBox = true;
      box = cvRect( x, y, 0, 0 );
      break;
    case CV_EVENT_LBUTTONUP:
      drawingBox = false;
      if(box.width < 0 )
	{
	  box.x += box.width;
	  box.width *= -1;
	}
      if( box.height < 0 )
	{
	  box.y += box.height;
	  box.height *= -1;
	}
      //st.loadSampleFromFile();
      st.getRoi();
        break;
    }
}
//绘制截图
void SvmTrain::getRoi()
{
  if( box.width==0||box.height==0)return;
  //只保留感兴趣图像部分
  tmp = cvCreateImageHeader(cvSize(box.width,box.height),src->depth,src->nChannels);
  tmp->origin = src->origin;
  tmp->widthStep = src->widthStep;
  //cvSetImageROI(src,box);
  tmp->imageData = src->imageData + box.y*src->widthStep + box.x * src->nChannels; 
  //tmp = cvCloneImage(src);
  //cvSetImageROI(tmp,box);
  cout<<src->width<<"  "<<tmp->width<<endl;
  //cvRectangle(tmp,cvPoint(box.x,box.y),cvPoint(box.x+box.width,box.y+box.height),cvScalar(0,255,0));
  collectData(tmp);
}
void SvmTrain::collectData(IplImage* ROi)
{
  cvShowImage( "block", tmp );
  cvResize(tmp,trainImg);
  hog->compute(trainImg, res,Size(1,1), Size(0,0)); 
  cout<<"HOG "<<res.size()<<endl;
  //放入临时存储变量中
  for(vector<float>::iterator iter = res.begin();iter!=res.end();iter++)
    {
      t_hog.push_back(*iter);
    }
  //显示分类
  cout<<pClass<<endl;
  cat_res.push_back(pClass);
  cvReleaseImage( &tmp );
  //数量增加
  pNum += 1;
  cout<<"the number of pic is "<<pNum<<endl;
  process_Train();
}
//训练图片归类
void SvmTrain::process_Train()
{ 
  if(cat_res.size()==1)return;
  data_mat = cvCreateMat(pNum,1764,CV_32FC1 );
  res_mat = cvCreateMat(pNum,1,CV_32FC1 );
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
      cvmSet( res_mat, i, 0, cat_res[i] ); 
    }
  cout<<"start to trainning"<<endl;
  svm.train(data_mat,res_mat,NULL,NULL,param);
  svm.save("thand.xml");
}
//清理垃圾，没有重写析构函数
void SvmTrain::clearMem()
{
  destroyAllWindows();
}
int main(int argc,char** argv)
{
  
  if(argc==1)
    SvmTrain tsv;
  else
    SvmTrain sv( atoi(argv[1]) );
  return 0;
}
