#include<iostream>
//#include<stdio.h>
#include<cv.h>
#include<highgui.h>
#include<ml.h>
#include<vector>
using namespace std;
using namespace cv;

class SvmTrain
{
public:
  SvmTrain();
private:
  //attributes
  vector<double> res;
  //methods
  //从文件中载入
  void loadFile();
  //从摄像头中获取训练样本
  void getSampleFromCAM();
  
};
SvmTrain::SvmTrain()
{
  cout<<"holy shit"<<endl;
  getchar();
}
int main()
{
  SvmTrain sv;
  return 0;
  int i,j,ii,jj;
  int width = 28,height = 30;
  int image_dim = width * height;
  int pnum = 10;
  IplImage *img_org,*sample_img;
  int res[10];
  float data[pnum*image_dim];
  CvMat data_mat,res_mat;
  CvSVM svm = CvSVM();
  CvTermCriteria criteria;
  CvSVMParams param;
  char filename[2];
  for( i=0;i<pnum;i++)
    {
      sprintf(filename,"handp/p%d/%d%d.png",i+1,i+1,i+1);
      cout<<filename<<" ";
      img_org = cvLoadImage(filename,CV_LOAD_IMAGE_GRAYSCALE);
      sample_img = cvCreateImage(cvSize(width,height),IPL_DEPTH_8U,1);
      cvResize(img_org,sample_img);
      cvSmooth(sample_img,sample_img,CV_GAUSSIAN,3,0,0,0);
      for(ii=0;ii<height;ii++)
	{
	  for(jj=0;jj<width;jj++)
	    {
	      data[ i * image_dim+(ii*width)+jj]=float((int)((uchar)(sample_img->imageData[ii*sample_img->widthStep+jj]))/255.0);
	    }
	}
      res[i]=(i+1)%2;
      cout<<res[i]<<endl;
    }

  cvInitMatHeader(&data_mat,pnum,image_dim,CV_32FC1,data);
  cvInitMatHeader(&res_mat,pnum,1,CV_32FC1,res);
  criteria = cvTermCriteria(CV_TERMCRIT_EPS,100,FLT_EPSILON);
  param = CvSVMParams(CvSVM::C_SVC,CvSVM::RBF, 10.0, 8.0, 1.0, 10.0, 0.02, 1.0,NULL,criteria);
  cout<<"--------------------"<<endl;
  svm.train(&data_mat,&res_mat,NULL,NULL,param);
  svm.save("svm_image.xml");
  //cvReleaseImage(&img_org);
  //cvReleaseImage(&sample_img);
  cout<<"===========================================predict"<<endl;
  img_org = cvLoadImage("handp/p2/2.png",CV_LOAD_IMAGE_GRAYSCALE);
  //svm.load("svm_image.xml");
  CvMat m ;
  cvInitMatHeader(&m,1,image_dim,CV_32FC1,NULL);
  IplImage *src_tmp = cvCreateImage(cvSize((int)(img_org->width/1.2),(int)(img_org->height/1.2)),IPL_DEPTH_8U,1);
  cvResize(img_org,src_tmp);
  
  ii=src_tmp->width;
  jj=src_tmp->height;
  float a[image_dim];
  for(i=0;i<=src_tmp->height-height;i+=3)
    {
      for(j=0;j<=src_tmp->width-width;j+=3)
	{
	  for(ii=0;ii<height;ii++)
	    for(jj=0;jj<width;jj++)
	      a[ii*width+j]=float((int)((uchar)(src_tmp->imageData[(ii+i)*src_tmp->widthStep+(jj+j)]))/255.0);
	}
    }
  cvSetData(&m,a,sizeof(float)*image_dim);
  float ret = -21.0;
  ret = svm.predict(&m);
  
  cout<<"the res is "<<ret<<endl;
  return 0;
}
