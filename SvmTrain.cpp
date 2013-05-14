#include<iostream>
//#include<stdio.h>
#include<cv.h>
#include<highgui.h>
#include<ml.h>

using namespace std;
using namespace cv;

class SvmTrain
{
  
};

int main()
{
  int i,j,ii,jj;
  int width = 28,height = 30;
  int image_dim = width * height;
  int pnum = 10;
  IplImage *img_org,*sample_img;
  int res[11];
  float data[pnum*image_dim];
  CvMat data_mat,res_mat;
  CvSVM svm = CvSVM();
  CvTermCriteria criteria;
  CvSVMParams param;
  char filename[2];
  cvNamedWindow("svm_org",CV_WINDOW_AUTOSIZE);
  cvShowImage("svm_org",cvLoadImage("1.bmp"));
  for( i=0;i<pnum;i++)
    {
      sprintf(filename,"%d.bmp",i+1,i+1);
      cout<<(img_org = cvLoadImage(filename,CV_LOAD_IMAGE_GRAYSCALE))<<endl;
      //cvShowImage("svm_org",img_org);
      //cout<<"load "<<filename<<endl;
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
      res[i]=1;
      cout<<img_org<<"      "<<filename<<endl;
      getchar();
    }
  cvInitMatHeader(&data_mat,pnum,image_dim,CV_32FC1,data);
  cvInitMatHeader(&res_mat,pnum,1,CV_32FC1,res);
  criteria = cvTermCriteria(CV_TERMCRIT_EPS,1000,FLT_EPSILON);
  param = CvSVMParams(CvSVM::NU_SVR,CvSVM::RBF,10.0,0.09,1.0,10.0,0.2,1.0,NULL,criteria);
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
