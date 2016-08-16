#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "assert.h"
#include "time.h"
#include "stdlib.h"
#include <math.h>
#include <algorithm>
#include "VideoSnapCut.h"
#define UNKNOWN_FLOW_THRESH 1e9 
//#include "opencv2/video/tracking.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include "lazysnapping.h"

extern "C" 
{
	#include "sift.h"
	#include "imgfeatures.h"
	#include "kdtree.h"
	#include "utils.h"
	#include "xform.h"
}
#include <cv.h>
#include <cxcore.h>
#include <highgui.h>

//template class Graph<double,double,double>;
//typedef Graph<double, double, double> GraphType;

int		VideoSnapCut::mMaskSize = 20;     //跟踪框的大小
int		VideoSnapCut::mMaskSize2 = VideoSnapCut::mMaskSize/2;
float	VideoSnapCut::mSigmaS = 5;
float	VideoSnapCut::mSigmaS2 = mSigmaS*mSigmaS;
int		VideoSnapCut::mK = 3;
int		VideoSnapCut::mMaxColorClassifiers = 4; 

VideoSnapCut::VideoSnapCut(void)
{
	mDataImagePixelClassifier = 0;	
}

VideoSnapCut::~VideoSnapCut(void)
{
	FreeColorClassifiers();
}

void VideoSnapCut::SetCurFrameMat(Mat CurFrameMat)
{
	mImage.release();
	CurFrameMat.copyTo(mImage);//修改，原来是等号赋值
	
	mImageF.release();	
	mImage.convertTo(mImageF,CV_32FC3,1.0 / 255.0);

	FreeDataImagePixelClassifier();
	mDataImagePixelClassifier = new DataImage<PixelClassifier>(mImageF.cols,mImageF.rows);//创建一组颜色分类器
}

void VideoSnapCut::SetCurFrame(int CurFrame)
{
	mCurFrame = CurFrame;
}

void VideoSnapCut::FreeColorClassifiers()
{
	DataImage<PixelClassifier>& dataImagePixelClassifier = *mDataImagePixelClassifier;
	for(uint i=0;i<mColorClassifiers.size();i++)
	{
		vector<ColorClassifier*>& colorClassifiers = mColorClassifiers[i]; //按照每个轮廓上的分类器遍历
		for(uint i=0;i<colorClassifiers.size();i++)
		{
			delete colorClassifiers[i];
		}		
		colorClassifiers.clear();
	}
	mColorClassifiers.clear();

	for(int r=0;r<(int)mImageF.rows;r++)    //将每一个点的分类器数量置零
	{
		for(int c = 0;c<(int)mImageF.cols;c++)
		{
			int& n = dataImagePixelClassifier(c,r).mNumClassifiers;
			n = 0;
		}//for(int r=0;r<(int)mImageF.rows;r++)
	}//for(int c = 0;c<(int)mImageF.cols;c++)
}

void VideoSnapCut::FreeDataImagePixelClassifier()
{
	if(mDataImagePixelClassifier)
	{
		delete mDataImagePixelClassifier;
	}
}

void VideoSnapCut::SetForegroundBorderPolygons(vector<PolygonF*>* ForegroundBorderPolygons)
{ 
	mForegroundBorderPolygons = ForegroundBorderPolygons;
}


int VideoSnapCut::BuildColorClassifiers()
{
	DataImage<PixelClassifier>& dataImagePixelClassifier = *mDataImagePixelClassifier;//存储分类器的
	
	FreeColorClassifiers();

	for(uint i=0;i<mSampledContourImages.size();i++)
	{
		Mat& sampledContourImages = mSampledContourImages[i];
		vector<ColorClassifier*> colorClassifiers;
		for(int r=0;r<(int)sampledContourImages.rows;r++)
		{
			for(int c = 0;c<(int)sampledContourImages.cols;c++)
			{
				int& pix = sampledContourImages.at<int>(r, c);//采样的时候已经将轮廓点的编号写入到sampleContourImages中了
				if(pix != -1)
				{
					//cout << "编号已经写入" << endl;
					ColorClassifier* colorClassifier = new ColorClassifier();
					
					colorClassifier->mImageF = mImageF;
					colorClassifier->mDataImagePixelClassifier = mDataImagePixelClassifier;
					colorClassifier->mCombinedMask = mCombinedMask;
					colorClassifier->mDistanceTransform = mDistanceTransforms[i];
					colorClassifier->mBoundingBoxCenter[0] = (float)c;
					colorClassifier->mBoundingBoxCenter[1] = (float)r;
					
					if(colorClassifier->Build())//每次都重新建立分类器
					{
						//cout << "建立分类器" << endl;
						pix = colorClassifiers.size();
						colorClassifiers.push_back(colorClassifier);
					}
					else //if(colorClassifier->Build())//当不足以建立分类器时编号置为-1
					{
						pix = -1;
						delete colorClassifier;
					}//else //if(colorClassifier->Build())
				}
			}//for(int c = 0;c<(int)sampledContourImages.cols;c++)
		}//for(int r=0;r<(int)sampledContourImages.rows;r++)
	
		mColorClassifiers.push_back(colorClassifiers);
		//DisplayBorderSamples(i);
	}//for(uint i=0;i<mSampledContourImages.size();i++)

	return 1; 
}

void VideoSnapCut::DisplayBorderSamples()
{
	int dist = VideoSnapCut::mMaskSize*2.0f/3;
	Mat borderAfter;
	for(int i = 0; i < mSampledContourImages.size(); i++)
	{
		Mat& sampledContourImages = mSampledContourImages[i];

		mImage.copyTo(borderAfter);
		for(int r=0;r<(int)borderAfter.rows;r++)
		{
			for(int c = 0;c<(int)borderAfter.cols;c++)
			{
				int pix = sampledContourImages.at<int>(r, c);
				if(pix != -1)
				{				
					Vec3b& intensity = borderAfter.at<Vec3b>(r, c);
					intensity.val[0] = 0;
					intensity.val[1] = 0;
					intensity.val[2] = 255;
					Point lu, rd;
					lu.x = max(c - dist, 0);
					lu.y = max(r - dist, 0);
					rd.x = min(c + dist, mImage.cols - 1);
					rd.y = min(r + dist, mImage.rows - 1);
					Rect r(lu, rd);			
					rectangle( borderAfter, lu, rd, Scalar(255,0,0), 1);		
				}			
			}
		}
	}
	imshow("afterContour",borderAfter);
}

void VideoSnapCut::DisplayBorderSamples(int i)
{
	char windowName[256];
	sprintf(windowName,"Sampled contour before %d",i+1);
		
	DisplayImage(windowName,mBorders[i]);
	Mat& sampledContourImages = mSampledContourImages[i];
	
	Mat borderAfter;	
	cvtColor(mBorders[i], borderAfter, CV_GRAY2RGB );
	borderAfter.convertTo(borderAfter,CV_8UC3,255);
	
	for(int r=0;r<(int)borderAfter.rows;r++)
	{
		for(int c = 0;c<(int)borderAfter.cols;c++)
		{
			int pix = sampledContourImages.at<int>(r, c);
			if(pix != -1)
			{				
				Vec3b& intensity = borderAfter.at<Vec3b>(r, c);
				intensity.val[0] = 0;
				intensity.val[1] = 0;
				intensity.val[2] = 255;
				/*Point lu, rd;
				lu.x = max(c - VideoSnapCut::mMaskSize2, 0);
				lu.y = max(r - VideoSnapCut::mMaskSize2, 0);
				rd.x = min(c + VideoSnapCut::mMaskSize2, mImage.cols - 1);
				rd.y = min(r - VideoSnapCut::mMaskSize2, mImage.rows - 1);
				Rect r(lu, rd);			
				rectangle( res, lu, rd, YELLOW, 1);		*/

			}			
		}
	}

	sprintf(windowName,"Sampled contour after %d",i+1);
	IplImage img = borderAfter;
	cvNamedWindow( windowName, 1 );
	cvShowImage( windowName, &img );
	char filename[256];
	sprintf(filename, "output\\%s.jpg", windowName);
	cvSaveImage(filename, &img);

	borderAfter.release();	

}
int VideoSnapCut::ClassifyPixels()
{
	DisplayImage("InputImage", mImage, 3, false);

	DataImage<PixelClassifier>& dataImagePixelClassifier = *mDataImagePixelClassifier;

	Mat maskImageProb(mCombinedMask.size(), CV_32FC1);


	Mat maskImage = mImage.clone();

	for(int r=0;r<(int)mCombinedMask.rows;r++)
	{
		for(int c = 0;c<(int)mCombinedMask.cols;c++)
		{
			maskImageProb.at<float>(r,c) = 0.0;
		}
	}

	uchar* tempData =  new uchar[mImageF.cols*mImageF.rows*3];
	int j = 0;
	for(int r=0;r<(int)mImageF.rows;r++)
	{
		for(int c = 0;c<(int)mImageF.cols;c++)
		{
			int numClassifiers = dataImagePixelClassifier(c,r).mNumClassifiers;
			Vec3f intensity = mImageF.at<Vec3f>(r, c);
			Color pix(intensity.val[2],intensity.val[1],intensity.val[0]);
			for(int i = 0;i<numClassifiers;i++)
			{
				//cout << "分类器数目不为0" << endl;
				ColorClassifier* colorClassifiers = dataImagePixelClassifier(c,r).mColorClassifier[i];

				float Pc_xF = colorClassifiers->mForegroundGMM->p(pix);
				float Pc_xB = colorClassifiers->mBackgroundGMM->p(pix);
				float Pc_x = Pc_xF / (Pc_xF + Pc_xB);

				dataImagePixelClassifier(c,r).mClassifierP_c_x[i] = Pc_x;		// probability

				float dXCen = (r-colorClassifiers->mBoundingBoxCenter[1])*(r-colorClassifiers->mBoundingBoxCenter[1])+
						(c-colorClassifiers->mBoundingBoxCenter[0])*(c-colorClassifiers->mBoundingBoxCenter[0]);
				dXCen = sqrt(dXCen);
				
				dataImagePixelClassifier(c,r).mClassifierWx[i] = 1.0f/(dXCen + Epsilon);

				float D_x = colorClassifiers->mDistanceTransform.at<float>(r,c);
				float D_x2 = D_x*D_x;
				mSigmaS2 = colorClassifiers->sigamS * colorClassifiers->sigamS;
				float F_x = 1 - exp(-D_x2/mSigmaS2);
				dataImagePixelClassifier(c,r).mClassifierF_s_x[i] = F_x;

			}
			int index1 = (r*mImageF.cols+c)*3;			
			
			pix.r *= 255; pix.r = pix.r < 0 ? 0 : pix.r > 255 ? 255 : pix.r;
			pix.g *= 255; pix.g = pix.g < 0 ? 0 : pix.g > 255 ? 255 : pix.g;
			pix.b *= 255; pix.b = pix.b < 0 ? 0 : pix.b > 255 ? 255 : pix.b;
			
			if(dataImagePixelClassifier(c,r).mNumClassifiers == 0)
			{
				if(mCombinedMask.at<uchar>(r, c)== 1)
				{

					tempData[index1+0] = 255;
					tempData[index1+1] = 255;
					tempData[index1+2] = 255;

					float t1 = (float)0.1, t2 = (float)(1-t1);
					tempData[index1+0] = (uchar)(t1*255 + t2*pix.b);
					tempData[index1+1] = (uchar)(t1*255 + t2*pix.g);
					tempData[index1+2] = (uchar)(t1*255 + t2*pix.r);

					//tempData[index1+0] = 0.9*pix.b;
					//tempData[index1+1] = 0.9*pix.g;
					//tempData[index1+2] = 0.9*pix.r;

					maskImageProb.at<float>(r,c) = 1.0;

					maskImage.at<Vec3b>(r, c) = Vec3b(pix.b, pix.g, pix.r);
				}
				else
				{
					tempData[index1+0] = 0;
					tempData[index1+1] = 0;
					tempData[index1+2] = 0;

					//float t1 = 0.5, t2 = 1-t1;
					//tempData[index1+0] = t1*0 + t2*pix.b;
					//tempData[index1+1] = t1*0 + t2*pix.g;
					//tempData[index1+2] = t1*0 + t2*pix.r;

					//tempData[index1+0] = 0.0*pix.b;
					//tempData[index1+1] = 0.0*pix.g;
					//tempData[index1+2] = 0.0*pix.r;
					maskImageProb.at<float>(r,c) = 0.0;

					maskImage.at<Vec3b>(r, c) = Vec3b(tempData[index1+0], tempData[index1+1], tempData[index1+2]);

				}
			}
			else
			{

				float Pf_x = 0;
				float Wf_x = 0;
				for(int i = 0;i<numClassifiers;i++)
				{
					int L_x = 0;
					if(mCombinedMask.at<uchar>(r, c) == 1)
					{
						L_x = 1;
					}
					float F_c_x = dataImagePixelClassifier(c,r).mClassifierF_s_x[i];
					float P_c_x = dataImagePixelClassifier(c,r).mClassifierP_c_x[i];
					float P_F_X = F_c_x * L_x + (1-F_c_x)*P_c_x;
					Wf_x += dataImagePixelClassifier(c,r).mClassifierWx[i];
					Pf_x += P_F_X*dataImagePixelClassifier(c,r).mClassifierWx[i];
				}

				Pf_x /= Wf_x;

				maskImageProb.at<float>(r,c) = Pf_x;
				/*cout << Pf_x <<",";
				if(++j%30 == 0)
					cout << endl;*/
				//	debug mClassifierF_s_x
				//Pf_x = dataImagePixelClassifier(c,r).mClassifierF_s_x[0];
				//int Pf_xColor = Pf_x * 255;
				//Pf_xColor = Pf_xColor < 0 ? 0 : Pf_xColor > 255 ? 255 : Pf_xColor;

				//	debug mF_C
				//int classifierID = dataImagePixelClassifier(c,r).mClassifierIDs[0];
				//ColorClassifier* classifier = mColorClassifiers[classifierID];
				//Pf_xColor = classifier->mF_C * 255;
				//Pf_xColor = Pf_xColor < 0 ? 0 : Pf_xColor > 255 ? 255 : Pf_xColor;

				//Pf_x *= 255; Pf_x = Pf_x < 0 ? 0 : Pf_x > 255 ? 255 : Pf_x;


				int Pf_xColor = (int)(Pf_x * 255);
				Pf_xColor = Pf_xColor < 0 ? 0 : Pf_xColor > 255 ? 255 : Pf_xColor;
				tempData[index1+0] = Pf_xColor;
				tempData[index1+1] = Pf_xColor;
				tempData[index1+2] = Pf_xColor;

				float t1 = 0.1f, t2 = (float)(1-t1);
				tempData[index1+0] = (float)(t1*Pf_xColor + t2*pix.b);
				tempData[index1+1] = (float)(t1*Pf_xColor + t2*pix.g);
				tempData[index1+2] = (float)(t1*Pf_xColor + t2*pix.r);

				maskImage.at<Vec3b>(r, c) = Vec3b(pix.b*Pf_x, pix.g*Pf_x, pix.r*Pf_x);

				//tempData[index1+0] = Pf_x*pix.b;
				//tempData[index1+1] = Pf_x*pix.g;
				//tempData[index1+2] = Pf_x*pix.r;
			}
			
		}
	}

	Mat maskImageProbChar = mCombinedMask.clone();

	for(int r=0;r<(int)mCombinedMask.rows;r++)
	{
		for(int c = 0;c<(int)mCombinedMask.cols;c++)
		{
			//float x = maskImageProb.at<float>(r,c);
			int pixelValue = (int)(maskImageProb.at<float>(r,c)*255);
			pixelValue = pixelValue<0?0:pixelValue>255?255:pixelValue;
			maskImageProbChar.at<uchar>(r,c) = pixelValue;
		}
	}

	mfPro = maskImageProb.clone ();    //将该权重作为全局混合高斯模型的
	////计算带权重的混合高斯模型
	Mat m, gcm;
	Mat fgdM, bgdM;
	GMM fgd(fgdM);
	GMM bgd(bgdM);
	mGlobalGMM(mImage, binMask1, mfPro, fgdM, bgdM);
	////waitKey(0);
	
	BandCut(mImage, fgdM, mfPro);
	//DisplayImage("窄带初始mask", binMask1);
	fgdM.copyTo(m);
	Mat m1, gcm1;
	BandCut1(mImage, m1, mfPro);
	Point p;
	int count = 0;
	Mat chaJi(mImage.size(), CV_8UC1);
	for(p.y = 0; p.y < mImage.rows; p.y++)
	{
		for(p.x = 0; p.x < mImage.cols; p.x++)
		{
			if(m.at<uchar>(p) != m1.at<uchar>(p))
			{
				chaJi.at<uchar>(p) = m.at<uchar>(p)^m1.at<uchar>(p);
				count++;
			}
		}
	}
	cout << "加入带权混合高斯前后分割的不同点数" << count << endl;
	m.copyTo(binMask);
	mImage.copyTo (gcm, m);
	mImage.copyTo(gcm1, m1);
	mGrabcutResult = m.clone();//将分割结果保存成01矩阵作为下一帧的初始轮廓
	DisplayImage("带权窄带grabcutResult", gcm, 3, false);
	DisplayImage("不带权窄带分割", gcm1, 3, false);
	DisplayImage("带权分割前后的差集", chaJi);

	char* name = new char[100];
	char* path = "E:\\shiyanResult\\第%d帧带权窄带grabcutResult.jpg";
	sprintf(name, path, getFrameNum());
	imwrite(name, gcm);
	delete[] name;

	char* name1 = new char[100];
	char* path1 = "E:\\shiyanResult\\第%d帧不带权窄带分割.jpg";
	sprintf(name1, path1, getFrameNum());
	imwrite(name1, gcm1);
	delete[] name1;
	
	char* name2 = new char[100];
	char* path2 = "E:\\shiyanResult\\第%d帧带权分割前后的差集.jpg";
	sprintf(name2, path2, getFrameNum());
	imwrite(name2, chaJi);
	delete[] name2;
	//waitKey();
	nextContourMat = m&1;

	//DisplayImage("ProbilityMap",maskImageProbChar,1, false);
	
	//DisplayImage("MaskOutputImage",maskImage, 3, false);

	//((CApp*)AfxGetApp())->mRightView->SetImageData(mImageF.cols, mImageF.rows, tempData, false,true);		
	//delete []tempData;
	
	return 1;

}
///////光流显示接口/////
void VideoSnapCut::makecolorwheel(vector<Scalar> &colorwheel)  
{  
    int RY = 15;  
    int YG = 6;  
    int GC = 4;  
    int CB = 11;  
    int BM = 13;  
    int MR = 6;  
  
    int i;  
  
    for (i = 0; i < RY; i++) colorwheel.push_back(Scalar(255,       255*i/RY,     0));  
    for (i = 0; i < YG; i++) colorwheel.push_back(Scalar(255-255*i/YG, 255,       0));  
    for (i = 0; i < GC; i++) colorwheel.push_back(Scalar(0,         255,      255*i/GC));  
    for (i = 0; i < CB; i++) colorwheel.push_back(Scalar(0,         255-255*i/CB, 255));  
    for (i = 0; i < BM; i++) colorwheel.push_back(Scalar(255*i/BM,      0,        255));  
    for (i = 0; i < MR; i++) colorwheel.push_back(Scalar(255,       0,        255-255*i/MR));  
}  
  
void VideoSnapCut::motionToColor(Mat flow, Mat &color)  
{  
    if (color.empty())  
        color.create(flow.rows, flow.cols, CV_8UC3);  
  
    static vector<Scalar> colorwheel; //Scalar r,g,b  
    if (colorwheel.empty())  
        makecolorwheel(colorwheel);  
  
    // determine motion range:  
    float maxrad = -1;  
  
    // Find max flow to normalize fx and fy  
    for (int i= 0; i < flow.rows; ++i)   
    {  
        for (int j = 0; j < flow.cols; ++j)   
        {  
            Vec2f flow_at_point = flow.at<Vec2f>(i, j);  
            float fx = flow_at_point[0];  
            float fy = flow_at_point[1];  
            if ((fabs(fx) >  UNKNOWN_FLOW_THRESH) || (fabs(fy) >  UNKNOWN_FLOW_THRESH))  
                continue;  
            float rad = sqrt(fx * fx + fy * fy);  
            maxrad = maxrad > rad ? maxrad : rad;  
        }  
    }  
  
    for (int i= 0; i < flow.rows; ++i)   
    {  
        for (int j = 0; j < flow.cols; ++j)   
        {  
            uchar *data = color.data + color.step[0] * i + color.step[1] * j;  
            Vec2f flow_at_point = flow.at<Vec2f>(i, j);  
  
            float fx = flow_at_point[0] / maxrad;  
            float fy = flow_at_point[1] / maxrad;  
            if ((fabs(fx) >  UNKNOWN_FLOW_THRESH) || (fabs(fy) >  UNKNOWN_FLOW_THRESH))  
            {  
                data[0] = data[1] = data[2] = 0;  
                continue;  
            }  
            float rad = sqrt(fx * fx + fy * fy);  
  
            float angle = (float)(atan2(-fy, -fx) / CV_PI);  
            float fk = (float)((angle + 1.0) / 2.0 * (colorwheel.size()-1));  
            int k0 = (int)fk;  
            int k1 = (k0 + 1) % colorwheel.size();  
            float f = fk - k0;  
            //f = 0; // uncomment to see original color wheel  
  
            for (int b = 0; b < 3; b++)   
            {  
                float col0 = (float)(colorwheel[k0][b] / 255.0);  
                float col1 = (float)(colorwheel[k1][b] / 255.0);  
                float col = (1 - f) * col0 + f * col1;  
                if (rad <= 1)  
                    col = 1 - rad * (1 - col); // increase saturation with radius  
                else  
                    col *= .75; // out of range  
                data[2 - b] = (int)(255.0 * col);  
            }  
        }  
    }  
} 

void VideoSnapCut::SetCurFrameMatOnly(Mat CurFrameMat)
{
	mImage.release();
	mImage = CurFrameMat;
	
	mImageF.release();	
	mImage.convertTo(mImageF,CV_32FC3,1.0 / 255.0);
}

void VideoSnapCut::RegisterCurFrameWithFrame(const Mat& iFrame)
{
	//	get frame I(t)
	//IplImage tempImg1 = mImage;
	IplImage img1 = mImage;//上一帧
	
	//	get frame I(t+1)
	IplImage img2 = iFrame;
	Mat nextFrameMat = iFrame;//表示下一帧
	//if(mMediaMode == 2)
	//{
	//	mVideoReader->GetFrame(iFrame, nextFrameMat);
	//	img2 = nextFrameMat;
	//}
	//else if(mMediaMode == 3)
	//{
	//	string path = (*mFilesPath)[iFrame];
	//	nextFrameMat = imread(path.data());
	//	img2 = nextFrameMat;
	//}


	//////计算前后两帧的光流/////

	/*Mat prevgray1, gray1, flow1, cflow1, frame1;
	cvtColor(mImage, prevgray1, COLOR_BGR2GRAY);
	cvtColor(nextFrameMat, gray1, COLOR_BGR2GRAY);
	calcOpticalFlowFarneback(prevgray1, gray1, flow1, 0.5, 3, 15, 3, 5, 1.2, 0);
	Mat motion2color1;
	motionToColor(flow1, motion2color1);  
    imshow("前一帧和后一帧的光流", motion2color1);*/
	//waitKey(0);
	////calcOpticalFlowSF(mImage, nextFrameMat, flow, 3, 2, 4, 4.1, 25.5, 18, 55.0, 25.5, 0.35, 18, 55.0, 25.5, 10);
	////光流计算完毕,存放在flow矩阵中////

	struct feature* feat1, * feat2, * feat;//图像特征
	struct feature** nbrs;
	struct kd_node* kd_root;
	CvPoint pt1, pt2;
	double d0, d1;
	int n1, n2, k, i, m = 0;

	IplImage* stacked = stack_imgs( &img1, &img2 );//合并两张图像
	n1 = sift_features( &img1, &feat1 );//获取shift特征点和特征点数（该函数是sift提供）
	n2 = sift_features( &img2, &feat2 );
	
	//	select features inside the object
	int n1In = 0;
	//DisplayImage("计算里面sift点", mCombinedMask);
	//imwrite("E:\\test\\result\\mConbinedMask.jpg",mCombinedMask);

	for( i = 0; i < n1; i++ )
	{
		feat = feat1 + i;
		int x = (int)feat->x;
		int y = (int)feat->y;
		if(mCombinedMask.at<uchar>(y, x) == 1)
		{
			feat1[n1In] = (*feat);
			n1In++;
		}
	}

	//fprintf( stderr, "Building kd tree...\n" );
	kd_root = kdtree_build( feat2, n2 ); ////sift提供
	
	for( i = 0; i < n1In; i++ )
	{
		feat = feat1 + i;
		k = kdtree_bbf_knn( kd_root, feat, 2, &nbrs, KDTREE_BBF_MAX_NN_CHKS );
		if( k == 2 ) 
		{
			d0 = descr_dist_sq( feat, nbrs[0] );
			d1 = descr_dist_sq( feat, nbrs[1] );
			if( d0 < d1 * NN_SQ_DIST_RATIO_THR )
			{
				pt1 = cvPoint( cvRound( feat->x ), cvRound( feat->y ) );
				pt2 = cvPoint( cvRound( nbrs[0]->x ), cvRound( nbrs[0]->y ) );
				pt2.y += img1.height;
				cvLine( stacked, pt1, pt2, CV_RGB(255,0,255), 1, 8, 0 );
				m++;
				feat1[i].fwd_match = nbrs[0];
			}
		}
		free( nbrs );
	}

	//fprintf( stderr, "Found %d total matches\n", m );
	//display_big_img( stacked, "Matches" );      //两张匹配的图像
	//cvWaitKey( 0 );


	//UNCOMMENT BELOW TO SEE HOW RANSAC FUNCTION WORKS
	//Note that this line above:
	//feat1[i].fwd_match = nbrs[0];
	//is important for the RANSAC function to work.

	float matData[9] = {1,0,0, 0,1,0, 0,0,1};
	Mat transform(3, 3, CV_32FC1, matData);

	if(n1In>0)
	{
		CvMat* xfrom;
		xfrom = ransac_xform( feat1, n1In, FEATURE_FWD_MATCH, lsq_homog, 4, 0.01, homog_xfer_err, 3.0, NULL, NULL );
		if( xfrom )
		{	
			transform = cvarrToMat(xfrom);      //就是为了获得这个变换矩阵
		}
	}


	CvSize size;
	size.width = mImageF.cols;
	size.height = mImageF.rows;

	Mat mImageT1;//原图像的仿射变换
	warpPerspective(mImage, mImageT1, transform, size, CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS, BORDER_CONSTANT,cvScalarAll( 0 ));
	//仿射变换后的图像
	//imshow("仿射变换后的图像", mImageT1);
	//waitKey(0);
	mImage.release();
	mImage = nextFrameMat.clone();

	//////计算仿射变换后的图像与第2帧图像的光流/////
	/*Mat prevgray, gray, flow, cflow, frame;
	cvtColor(mImageT1, prevgray, COLOR_BGR2GRAY);
	imshow("仿射变换后的图像", mImageT1);
	cvtColor(iFrame, gray, COLOR_BGR2GRAY);
	imshow("下一帧图像", iFrame);
	calcOpticalFlowFarneback(prevgray, gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
	Mat motion2color;
	motionToColor(flow, motion2color);  
    imshow("前一帧仿射变换前后的光流", motion2color);*/
	//waitKey(0);
	////calcOpticalFlowSF(mImage, nextFrameMat, flow, 3, 2, 4, 4.1, 25.5, 18, 55.0, 25.5, 0.35, 18, 55.0, 25.5, 10);
	//////光流计算完毕,存放在flow矩阵中////


	//DisplayImage("Transformed int image",mImageT1, 3,false);
	
	Mat mImageFT1;	
	warpPerspective(mImageF, mImageFT1, transform, size, CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS, BORDER_CONSTANT,cvScalarAll( 0 ));
	
	mImageF.release();	
	mImage.convertTo(mImageF,CV_32FC3,1.0 / 255.0);

	DisplayImage("Transformed float image",mImageFT1,3,false);
	//记录仿射变换后的图像
	//Mat clonedMat = mImageFT1.clone();
	//IplImage img = clonedMat;
	//for( int y=0; y<img.height; y++ ) 
	//{ 
	//    uchar* dest = (uchar*) ( img.imageData + y * img.widthStep ); 
	//					
	//	for( int x=0; x<img.width; x++ ) 
	//	{ 		
	//		for(int c=0;c<3;c++)
	//		{
	//			if(1)
	//			{
	//				dest[x*3 + c] *= 255;//dest里面存的就是是小0~1的像素值
	//			}
	//		}
	//		
	//	}
	//}
	//cvSaveImage("E:\\shiyanResult\\Transformed.jpg", &img);
	Mat mImageFTcopy;
	mImageFT1.copyTo(mImageFTcopy);
	Point p;
	for(p.y = 0; p.y < mImageFT1.rows; p.y++)
	{
		for(p.x = 0; p.x < mImageFT1.cols; p.x++)
		{
			mImageFTcopy.at<float>(p) = mImageFTcopy.at<float>(p)*255 < 255 ? mImageFTcopy.at<float>(p)*255 : 255 ;
		}
	}
	imwrite("E:\\shiyanResult\\Transformed.jpg", mImageFTcopy);
	mAfterTransContours.clear();
	for(uint i=0;i<mMasks.size();i++)
	{
		char windowName[256];
		
		CvSize size;
		size.width = mMasks[i].cols;
		size.height = mMasks[i].rows;
		
		Mat maskT1;

		//DisplayImage("仿射变换前轮廓", mMasks[i]);
		warpPerspective(mMasks[i], maskT1, transform, size, CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS, BORDER_CONSTANT,cvScalarAll( 0 ));
		//DisplayImage("仿射变换后轮廓", maskT1);
		mMasks[i].release();
		mMasks[i] = maskT1;

		sprintf(windowName,"Transformed mask %d",i+1);
		//DisplayImage(windowName,maskT1);

		Mat borderT1 = CreateContour(maskT1);
		mAfterTransContours.push_back(borderT1);
		//DisplayImage("变换后的轮廓线",borderT1);
		mBorders[i].release();
		mBorders[i] = borderT1;

		sprintf(windowName,"Transformed border %d",i+1);
		//DisplayImage(windowName,borderT1);

		Mat distanceTransformT1 = CreateDistanceTransforms(borderT1);
		mDistanceTransforms[i].release();
		mDistanceTransforms[i] = distanceTransformT1;

		sprintf(windowName,"Transformed distance transform %d",i+1);
		//DisplayDistanceImage(windowName,distanceTransformT1);

		Mat sampledContourImageT1;		
		warpPerspective(mSampledContourImages[i], sampledContourImageT1, transform, size, CV_INTER_NN + CV_WARP_FILL_OUTLIERS, BORDER_CONSTANT,cvScalarAll( -1 ));
		
		mSampledContourImages[i].release();
		mSampledContourImages[i] = sampledContourImageT1;
		//DisplayBorderSamples(i);
	}

	Mat combinedMaskT1;
	warpPerspective(mCombinedMask, combinedMaskT1, transform, size, CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS, BORDER_CONSTANT,cvScalarAll( 0 ));
	//DisplayImage("Transformed combined mask",combinedMaskT1);
	mCombinedMask.release();
	mCombinedMask = combinedMaskT1;

	//BuildColorClassifiers();
	mCurFrame++;
}

void VideoSnapCut::ClearMasks()
{
	for(uint i=0;i<mMasks.size();i++)
	{
		mMasks[i].release();
	}
	mMasks.clear();

	mCombinedMask.release();

	for(uint i=0;i<mBorders.size();i++)
	{
		mBorders[i].release();
	}
	mBorders.clear();

	for(uint i=0;i<mDistanceTransforms.size();i++)
	{
		mDistanceTransforms[i].release();
	}
	mDistanceTransforms.clear();

	for(uint i=0;i<mSampledContourImages.size();i++)
	{
		mSampledContourImages[i].release();
	}
	mSampledContourImages.clear();
}

bool VideoSnapCut::PointInPolygon(const PolygonF* polygon, float y, float x) //判断点在轮廓里还是轮廓外
{
	bool oddNodes=false;
		
	int polygonSize = polygon->mPoints.size();//轮廓上点的数量
	int      i, j=polygonSize-1;
	
	for (i=0; i<polygonSize; i++) 
	{    
		if ((polygon->mPoints[i].y< y && polygon->mPoints[j].y>=y 
			|| polygon->mPoints[j].y< y && polygon->mPoints[i].y>=y) 
			&& (polygon->mPoints[i].x<=x || polygon->mPoints[j].x<=x)) 
		{
			oddNodes^=
				(polygon->mPoints[i].x+(y-polygon->mPoints[i].y)/
				(polygon->mPoints[j].y-polygon->mPoints[i].y)*
				(polygon->mPoints[j].x-polygon->mPoints[i].x)<x); 
		}
		j=i; 
	}

	return oddNodes;
}

void VideoSnapCut::DisplayImage(char* name, const Mat& mat, int channels/* = 1*/, bool multiply/* = true*/)
{
	Mat clonedMat = mat.clone();
	IplImage img = clonedMat;
	for( int y=0; y<img.height; y++ ) 
	{ 
		uchar* dest = (uchar*) ( img.imageData + y * img.widthStep ); 
						
		for( int x=0; x<img.width; x++ ) 
		{ 		
			for(int c=0;c<channels;c++)
			{
				if(multiply)
				{
					dest[x*channels + c] *= 255;//dest里面存的就是是小0~1的像素值
				}
			}
			
		}
	}
	cvNamedWindow( name, 1 );
	cvShowImage( name, &img );
	char filename[256];
	sprintf(filename, "output\\%s.jpg", name);
	cvSaveImage(filename, &img);
	clonedMat.release();
}

void VideoSnapCut::DisplayDistanceImage(char* name, const Mat& mat, int channels/* = 1*/, bool multiply/* = true*/)
{
	float maxDist = 0;
	Mat clonedMat = mat.clone();
	IplImage img = clonedMat;

	for( int y=0; y<img.height; y++ ) 
	{ 
		float* dest = (float*) ( img.imageData + y * img.widthStep ); 
						
		for( int x=0; x<img.width; x++ ) 
		{ 		
			if(dest[x]>maxDist)
			{
				maxDist = dest[x];
			}		
			
		}
	}

	for( int y=0; y<img.height; y++ ) 
	{ 
		float* dest = (float*) ( img.imageData + y * img.widthStep ); 
						
		for( int x=0; x<img.width; x++ ) 
		{ 		
			dest[x] /= maxDist;
			dest[x]  = dest[x]*0.9f + 1*0.1f; 			
		}
	}
	cvNamedWindow( name, 1 );
	cvShowImage( name, &img );

	for( int y=0; y<img.height; y++ ) 
	{ 
		float* dest = (float*) ( img.imageData + y * img.widthStep ); 
						
		for( int x=0; x<img.width; x++ ) 
		{ 		
			dest[x] *= 255;			
		}
	}

	char filename[256];
	sprintf(filename, "output\\%s.jpg", name);
	cvSaveImage(filename, &img);

	clonedMat.release();
}


void VideoSnapCut::CreateMasks()
{
	ClearMasks();
	vector<PolygonF*>&		foregroundBorderPolygons =  *mForegroundBorderPolygons;//初始化轮廓线

	char windowName[256];
	CvSize size;
	
	size.width = mImageF.cols;//调用createmask时必须先初始化mImageF，导入当前帧时初始化
	size.height = mImageF.rows;

	mCombinedMask.create(size, CV_8UC1);//每次调用CreateMasks时就要初始化combinedMask
	mCombinedMask = mCombinedMask.zeros(size, CV_8UC1);

	for(uint ii=0;ii<foregroundBorderPolygons.size();ii++)
	{
		int polygonSize = foregroundBorderPolygons[ii]->mPoints.size();
		if(polygonSize<2)
		{
			continue;
		}

		PolygonF* polygon = foregroundBorderPolygons[ii];
		
		Mat mask = CreateMask(polygon);//二值图,表示轮廓矩阵0表示背景1表示前景
		mMasks.push_back(mask);        //每一个轮廓作为一个矩阵存入mMasks中
		
		sprintf(windowName,"Mask %d",ii);
		//DisplayImage(windowName, mask);	//显示的矩阵都是mat型的	

		Mat border = CreateContour(mask);//创建一个轮廓0,1没有对点进行编号
		mBorders.push_back(border);
		
		sprintf(windowName,"border %d",ii);
		//DisplayImage(windowName, border);
		////沿轮廓形成窄带
		//Mat borderB = CreateContourBand(mask);
		//mBand = borderB;
		//DisplayImage("borderBand", borderB);

		Mat distanceTransform = CreateDistanceTransforms(border);
		mDistanceTransforms.push_back(distanceTransform);


		sprintf(windowName,"distanceTransforms %d",ii);
		//DisplayDistanceImage(windowName, distanceTransform,1,false);

		Mat sampledContourImage = CreatSampleContourImage(mImageF);	////每个元素都为-1	
		mSampledContourImages.push_back(sampledContourImage);

	}//for(int ii=0;ii<mForegroundBorderPolygons->size();ii++)

	
	sprintf(windowName,"Combined Mask");
	//DisplayImage(windowName, mCombinedMask);

	SampleContour();    // 之前的过程已经均匀化过了，忘记初始化采样点编号了	
}

Mat VideoSnapCut::CreateMask(const PolygonF* Polygon)//mask是CV_8UC1的单通道矩阵
{
	CvSize size;
	size.width = mImageF.cols;
	size.height = mImageF.rows;

	Mat mask(size, CV_8UC1);
			
	for(int y=0;y<mask.rows;y++)
	{
		for(int x = 0;x<mask.cols;x++)
		{
			mask.at<uchar>(y, x) = PointInPolygon(Polygon,(float)y,(float)x);
			//mCombinedMask.at<uchar>(y, x) ^= mask.at<uchar>(y, x);//mConbinedMask也改变了
			mCombinedMask.at<uchar>(y,x) = mask.at<uchar>(y,x);
		}
	}
	return mask;
}

Mat VideoSnapCut::CreateContour(const Mat& Mask)//通过Mask矩阵生成轮廓线矩阵
{
	CvSize size;
	size.width = Mask.cols;
	size.height = Mask.rows;

	Mat border(size, CV_8UC1);
			
	for(int y=0;y<border.rows;y++)
	{
		for(int x = 0;x<border.cols;x++)
		{
			bool borderPixel = false;
			int xx,yy;
			if(Mask.at<uchar>(y, x) == 1) // foreground
			{
				xx = x - 1;
				yy = y    ;
				if(xx==-1 || (xx >= 0 && !Mask.at<uchar>(yy, xx)))
				{
					borderPixel = true;						
				}
				
				xx = x + 1;
				yy = y    ;
				if(xx==border.cols || ( xx < border.cols && !Mask.at<uchar>(yy, xx) ))
				{
					borderPixel = true;						
				}
				
				xx = x	  ;	
				yy = y - 1;
				if(yy==-1 || (yy >= 0 && !Mask.at<uchar>(yy, xx)))
				{
					borderPixel = true;						
				}
				
				xx = x    ;
				yy = y + 1;
				if(yy==border.rows || (yy < border.rows && !Mask.at<uchar>(yy, xx)))
				{
					borderPixel = true;						
				}					
			}
			border.at<uchar>(y, x) = borderPixel;
		}
	}
	return border;

}


Mat VideoSnapCut::CreateDistanceTransforms(const Mat& Border)
{
	CvSize size;
	size.width = Border.cols;
	size.height = Border.rows;

	float dx = 1;
	float dy = 1;
	float dxy = sqrt(2.0f);

	Mat distanceTransforms(size, CV_32FC1);

	// clear 
	for(int y=0;y<distanceTransforms.rows;y++)
	{
		for(int x = 0;x<distanceTransforms.cols;x++)
		{
			if(Border.at<uchar>(y, x))
			{
				distanceTransforms.at<float>(y, x) = 0;
			}
			else
			{
				distanceTransforms.at<float>(y, x) = INF;
			}
		}
	}

	// forward pass
	for(int y=0;y<distanceTransforms.rows;y++)
	{
		for(int x = 0;x<distanceTransforms.cols;x++)
		{
			if(y-1>=0)
			{	
				distanceTransforms.at<float>(y, x) = min(distanceTransforms.at<float>(y, x), 
					distanceTransforms.at<float>(y-1, x) + dy);
			}
			if(x-1>=0)
			{	
				distanceTransforms.at<float>(y, x) = min(distanceTransforms.at<float>(y, x), 
					distanceTransforms.at<float>(y, x-1) + dx);
			}
			if(x-1>=0 && y-1>=0)
			{
				distanceTransforms.at<float>(y, x) = min(distanceTransforms.at<float>(y, x), 
					distanceTransforms.at<float>(y-1, x-1) + dxy);
			}
		}
	}

	// backword pass
	for(int y=distanceTransforms.rows-1;y>=0;y--)
	{
		for(int x=distanceTransforms.cols-1;x>=0;x--)
		{
			if(y+1<Border.rows)
			{
				distanceTransforms.at<float>(y, x) = min(distanceTransforms.at<float>(y, x), 
					distanceTransforms.at<float>(y+1, x) + dy);
			}
			if(x+1<Border.cols)
			{
				distanceTransforms.at<float>(y, x) = min(distanceTransforms.at<float>(y, x), 
					distanceTransforms.at<float>(y, x+1) + dx);
			}
			if(x+1<Border.cols && y+1<Border.rows)
			{
				distanceTransforms.at<float>(y, x) = min(distanceTransforms.at<float>(y, x), 
					distanceTransforms.at<float>(y+1, x+1) + dxy);
			}
		}
	}
	return distanceTransforms;
}
void VideoSnapCut::ClearBorderSamples()
{
	for(uint i=0;i<mForegroundSampledBorderPolygons.size();i++)
	{
		delete mForegroundSampledBorderPolygons[i];
	}
	mForegroundSampledBorderPolygons.clear();
}

Mat VideoSnapCut::CreatSampleContourImage(const Mat& image)
{
	CvSize size;
	size.width = image.cols;
	size.height = image.rows;

	Mat sampledContourImage(size,CV_32SC1);
	for(int r=0;r<sampledContourImage.rows;r++)
	{
		for(int c = 0;c<sampledContourImage.cols;c++)
		{
			int& pix = sampledContourImage.at<int>(r, c);
			pix = -1;			
		}
	}
	return sampledContourImage;
}

void VideoSnapCut::SampleContour()//mSampledContourImage初始化的全是-1
{
	ClearBorderSamples();//清除前景轮廓向量

	for(uint i=0;i<mForegroundBorderPolygons->size();i++)
	{
		Mat& sampledContourImages = mSampledContourImages[i];
		int polygonSize = (*mForegroundBorderPolygons)[i]->mPoints.size();//轮廓中所有点的数目
		if(polygonSize<1)
		{
			continue;
		}
		PolygonF* polygon = (*mForegroundBorderPolygons)[i];//轮廓初始点
		
		PolygonF* sampledPolygon = new PolygonF();
		
		float curDist = 0;
		int curPolygonPointIndex = 1;//当前轮廓点索引
		float curMagnitude;
		float sampleDist = VideoSnapCut::mMaskSize*2.0f/3;//采样距离
		sampledPolygon->mPoints.push_back(polygon->mPoints[0]);//将polygon的第0个点推入采样轮廓集合里
		PointF prevPoint = polygon->mPoints[0];
		PointF curPoint;
		

		while(curPolygonPointIndex<=polygonSize)
		{
			int curPolygonPointIndexMod = curPolygonPointIndex % polygonSize;
			curPoint = polygon->mPoints[curPolygonPointIndexMod];
			curMagnitude = sqrt((curPoint.x - prevPoint.x)*(curPoint.x - prevPoint.x)+(curPoint.y - prevPoint.y)*(curPoint.y - prevPoint.y));
			if(curDist+curMagnitude<sampleDist)
			{
				curDist += curMagnitude;
				prevPoint = curPoint;
				curPolygonPointIndex++;

			}
			else
			{
				float remainingDist = sampleDist - curDist;
				float t = remainingDist / curMagnitude;
				float x0 = prevPoint.x;
				float y0 = prevPoint.y;
				float dx = curPoint.x - prevPoint.x;
				float dy = curPoint.y - prevPoint.y;
				PointF newPoint = PointF(x0 + t * dx, y0 + t * dy);
				
				int c = (int)newPoint.x;
				int r = (int)newPoint.y;
				int& pix = sampledContourImages.at<int>(r, c);
				pix = sampledPolygon->mPoints.size();

				sampledPolygon->mPoints.push_back(newPoint);

				prevPoint = newPoint;
				curDist = 0;

			} 
		}
		PointF fisrtPoint = sampledPolygon->mPoints[0];
		PointF lastPoint = sampledPolygon->mPoints[sampledPolygon->mPoints.size()-1];
		float dx = lastPoint.x - fisrtPoint.x;
		float dy = lastPoint.y - fisrtPoint.y;				
		float dist = sqrt(dx*dx + dy*dy);
		if(dist<sampleDist*1.0)
		{
			sampledPolygon->mPoints.pop_back();
		}
		mForegroundSampledBorderPolygons.push_back(sampledPolygon);
	}
	
}

void VideoSnapCut::generateContourWithRect(Mat& tmp)
{
	Mat con;
	tmp.copyTo(con);
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(con, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	vector<Point> poly; 
	vector<Rect> polyRect;
	
	int idx = 0;
	for(; idx >= 0; idx = hierarchy[idx][0])
	{
		approxPolyDP(Mat(contours[idx]), poly, 0.00001, true);
	}

	PolygonF* pyF = new PolygonF();
	for(int i = 0; i < poly.size(); i = i + 1)
	{

		PointF pt;
		pt.x = (float)(poly[i].x);
		pt.y = (float)(poly[i].y);
		pyF->mPoints.push_back(pt);
		
	}
	ClearBorderSamples();
	vector<PolygonF*>* p = new vector<PolygonF*>();
	p -> push_back(pyF);
	mForegroundBorderPolygons = p;//从画笔中获得的轮廓多边形
}

void VideoSnapCut::ResampleContour()
{
	ClearBorderSamples();

	for(uint i=0;i<mSampledContourImages.size();i++)
	{
		PolygonF* sampledPolygon = new PolygonF();

		Mat& sampledContourImages = mSampledContourImages[i];
		
		for(int r=0;r<(int)sampledContourImages.rows;r++)
		{
			for(int c = 0;c<(int)sampledContourImages.cols;c++)
			{
				int& pix = sampledContourImages.at<int>(r, c);
				if(pix != -1)
				{
					sampledPolygon->mPoints.push_back(PointF((float)c,(float)r));

				}
			}//for(int c = 0;c<(int)sampledContourImages.cols;c++)
		}//for(int r=0;r<(int)sampledContourImages.rows;r++)
	
		mForegroundSampledBorderPolygons.push_back(sampledPolygon);
		//DisplayBorderSamples(i);
	}//for(uint i=0;i<mSampledContourImages.size();i++)

}

//窄带分割相关函数实现
double VideoSnapCut::calcBeta(const Mat& img) {
    double beta = 0;
    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            Vec3d color = img.at<Vec3b>(y, x);
            if (x > 0) // left
                    {
                Vec3d diff = color - (Vec3d) img.at<Vec3b>(y, x - 1);
                beta += diff.dot(diff);
            }
            if (y > 0 && x > 0) // upleft
                    {
                Vec3d diff = color - (Vec3d) img.at<Vec3b>(y - 1, x - 1);
                beta += diff.dot(diff);
            }
            if (y > 0) // up
                    {
                Vec3d diff = color - (Vec3d) img.at<Vec3b>(y - 1, x);
                beta += diff.dot(diff);
            }
            if (y > 0 && x < img.cols - 1) // upright
                    {
                Vec3d diff = color - (Vec3d) img.at<Vec3b>(y - 1, x + 1);
                beta += diff.dot(diff);
            }
        }
    }
    if (beta <= std::numeric_limits<double>::epsilon())
        beta = 0;
    else
        beta = 1.f
                / (2 * beta
                        / (4 * img.cols * img.rows - 3 * img.cols - 3 * img.rows
                                + 2));

    return beta;
}

void VideoSnapCut::calcNWeights(const Mat& img, Mat& leftW, Mat& upleftW, Mat& upW,
        Mat& uprightW, double beta, double gamma) {
    const double gammaDivSqrt2 = gamma / std::sqrt(2.0f);
    leftW.create(img.rows, img.cols, CV_64FC1);
    upleftW.create(img.rows, img.cols, CV_64FC1);
    upW.create(img.rows, img.cols, CV_64FC1);
    uprightW.create(img.rows, img.cols, CV_64FC1);
    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            Vec3d color = img.at<Vec3b>(y, x);
            if (x - 1 >= 0) // left
                    {
                Vec3d diff = color - (Vec3d) img.at<Vec3b>(y, x - 1);
                leftW.at<double>(y, x) = gamma * exp(-beta * diff.dot(diff));
            } else
                leftW.at<double>(y, x) = 0;
            if (x - 1 >= 0 && y - 1 >= 0) // upleft
                    {
                Vec3d diff = color - (Vec3d) img.at<Vec3b>(y - 1, x - 1);
                upleftW.at<double>(y, x) = gammaDivSqrt2
                        * exp(-beta * diff.dot(diff));
            } else
                upleftW.at<double>(y, x) = 0;
            if (y - 1 >= 0) // up
                    {
                Vec3d diff = color - (Vec3d) img.at<Vec3b>(y - 1, x);
                upW.at<double>(y, x) = gamma * exp(-beta * diff.dot(diff));
            } else
                upW.at<double>(y, x) = 0;
            if (x + 1 < img.cols - 1 && y - 1 >= 0) // upright
                    {
                Vec3d diff = color - (Vec3d) img.at<Vec3b>(y - 1, x + 1);
                uprightW.at<double>(y, x) = gammaDivSqrt2
                        * exp(-beta * diff.dot(diff));
            } else
                uprightW.at<double>(y, x) = 0;
        }
    }
}
void VideoSnapCut::constructGCGraph(const Mat& img, const Mat& imgPro, double lambda, const Mat& leftW, const Mat& upleftW,
        const Mat& upW, const Mat& uprightW, GraphType* graph) {
    Point p;
    int vtxIdx=0;
    for (p.y = 0; p.y < img.rows; p.y++) {
        for (p.x = 0; p.x < img.cols; p.x++) {
            // add node
            Vec3b color = img.at<Vec3b>(p);

            // set t-weights
            //double fromSource, toSink;
            //if (mask.at<uchar>(p) == GC_BGD) {  // GC_BGD
            //    fromSource = 0;
            //    toSink = lambda;
            //} else if (mask.at<uchar>(p) == GC_FGD) // GC_FGD
            //{
            //    fromSource = lambda;
            //    toSink = 0;
            //} else
            //{
            //    fromSource = -log(bgdGMM(color));
            //    toSink = -log(fgdGMM(color));
            //}
			double  fromSource, toSink;
			if(imgPro.at<float>(p) < 0.1)
			{
				fromSource = 0;
                toSink = lambda;
			}
			else if(imgPro.at<float>(p) > 0.9)
			{
				fromSource = lambda;
                toSink = 0;
			}
			else{
				fromSource = - log((double)(imgPro.at<float>(p)));
			    toSink = - log((double)(1 - imgPro.at<float>(p)));
			}
			
            graph->add_node();
            graph->add_tweights(vtxIdx, fromSource, toSink);

            // set n-weights
            if (p.x > 0) {
                double w = leftW.at<double>(p);
                graph->add_edge(vtxIdx, vtxIdx - 1, w, w);
            }
            if (p.x > 0 && p.y > 0) {
                double w = upleftW.at<double>(p);
                graph->add_edge(vtxIdx, vtxIdx - img.cols - 1, w, w);
            }
            if (p.y > 0) {
                double w = upW.at<double>(p);
                graph->add_edge(vtxIdx, vtxIdx - img.cols, w, w);
            }
            if (p.x < img.cols - 1 && p.y > 0) {
                double w = uprightW.at<double>(p);
                graph->add_edge(vtxIdx, vtxIdx - img.cols + 1, w, w);
            }
            vtxIdx++;
        }
    }
}

void VideoSnapCut::estimateSegmentation(GraphType* graph, Mat& mask) {
    graph->maxflow();
    Point p;
    for (p.y = 0; p.y < mask.rows; p.y++) {
        for (p.x = 0; p.x < mask.cols; p.x++) {        
                if (graph->what_segment(p.y * mask.cols + p.x) == GraphType::SOURCE)
                    mask.at<uchar>(p) = 1;
                else
                    mask.at<uchar>(p) = 0;
            
        }
    }
}

void VideoSnapCut::BandCut(const Mat& img, Mat& mask, Mat& fgdPro)//采用局部分类器的概率作为triMap
{
	
	Mat bgdModel, fgdModel;
	GMM bgdGMM( bgdModel ), fgdGMM( fgdModel );
	initGMMs( img, mask, bgdGMM, fgdGMM );//将矩形框内设置为可能前景
    Mat compIdxs( img.size(), CV_32SC1 );
	const double gamma = 50;//50
	const double lamda = 450;//9*50;
	const double beta = calcBeta(img);
	Mat leftW, upleftW, upW, uprightW;
	calcNWeights(img, leftW, upleftW, upW, uprightW, beta, gamma);
	GraphType* graph = NULL;
	int vexCount = img.cols * img.rows;
	int edgeCount = 2 * (4 * img.cols *img.rows - 3 * (img.cols + img.rows) + 2);
	graph = new GraphType(vexCount, edgeCount);//初始化图结构
	assignGMMsComponents( img, mask, bgdGMM, fgdGMM, compIdxs );
    learnGMMs( img, mask, compIdxs, bgdGMM, fgdGMM );
	constructGCGraph(img, mask, bgdGMM, fgdGMM, lamda, leftW, upleftW, upW, uprightW, graph );
	Mat m(img.size(), CV_8UC1);
    estimateSegmentation(graph, m);
	//DisplayImage("ProbilityMapGrabcut", mask,1, false);
	//imshow("ProbilityMapGrabcut", m);
	if(graph) delete graph;
	mask = m.clone();
}

void VideoSnapCut::BandCut1(const Mat& img, Mat& mask, Mat& fgdPro)
{
	
	//Mat bgdModel, fgdModel;
	//GMM bgdGMM( bgdModel ), fgdGMM( fgdModel );
	//initGMMs( img, mask, bgdGMM, fgdGMM );//将矩形框内设置为可能前景
 //   Mat compIdxs( img.size(), CV_32SC1 );
	const double gamma = 50;//50
	const double lamda = 450;//9*50;
	const double beta = calcBeta(img);
	Mat leftW, upleftW, upW, uprightW;
	calcNWeights(img, leftW, upleftW, upW, uprightW, beta, gamma);
	GraphType* graph = NULL;
	int vexCount = img.cols * img.rows;
	int edgeCount = 2 * (4 * img.cols *img.rows - 3 * (img.cols + img.rows) + 2);
	graph = new GraphType(vexCount, edgeCount);//初始化图结构
	/*assignGMMsComponents( img, mask, bgdGMM, fgdGMM, compIdxs );
    learnGMMs( img, mask, compIdxs, bgdGMM, fgdGMM );
	constructGCGraph(img, mask, bgdGMM, fgdGMM, lamda, leftW, upleftW, upW, uprightW, graph );*/
	constructGCGraph(img, mfPro, lamda, leftW, upleftW, upW, uprightW, graph);
	Mat m(img.size(), CV_8UC1);
    estimateSegmentation(graph, m);
	//DisplayImage("ProbilityMapGrabcut", mask,1, false);
	//imshow("ProbilityMapGrabcut", m);
	if(graph) delete graph;
	mask = m.clone();
}
void VideoSnapCut::constructGCGraph( const Mat& img, const Mat& mask, const GMM& bgdGMM, const GMM& fgdGMM, double lambda,
                       const Mat& leftW, const Mat& upleftW, const Mat& upW, const Mat& uprightW,
                        GraphType* graph )
{
    /*int vtxCount = img.cols*img.rows,
        edgeCount = 2*(4*img.cols*img.rows - 3*(img.cols + img.rows) + 2);
    graph = new GraphType(vtxCount, edgeCount);*/
    Point p;
    for( p.y = 0; p.y < img.rows; p.y++ )
    {
        for( p.x = 0; p.x < img.cols; p.x++)
        {
            // add node
            int vtxIdx = graph -> add_node();//从左到右从上到下依次将点加入到图中
            Vec3b color = img.at<Vec3b>(p);

            // set t-weights
            double fromSource, toSink;
            if( mask.at<uchar>(p) == GC_PR_BGD || mask.at<uchar>(p) == GC_PR_FGD )
            {
                fromSource = -log( bgdGMM(color) );
                toSink = -log( fgdGMM(color) );
            }
            else if( mask.at<uchar>(p) == GC_BGD )
            {
                fromSource = 0;
                toSink = lambda;
            }
            else // GC_FGD
            {
                fromSource = lambda;
                toSink = 0;
            }
            graph -> add_tweights( vtxIdx, fromSource, toSink );//加入该点的t-link

            // set n-weights
            if( p.x>0 )
            {
                double w = leftW.at<double>(p);
                graph -> add_edge( vtxIdx, vtxIdx-1, w, w ); //根据当前点找到相邻点并将该边加入到图中
            }
            if( p.x>0 && p.y>0 )
            {
                double w = upleftW.at<double>(p);
                graph -> add_edge( vtxIdx, vtxIdx-img.cols-1, w, w );
            }
            if( p.y>0 )
            {
                double w = upW.at<double>(p);
                graph -> add_edge( vtxIdx, vtxIdx-img.cols, w, w );
            }
            if( p.x<img.cols-1 && p.y>0 )
            {
                double w = uprightW.at<double>(p);
                graph -> add_edge( vtxIdx, vtxIdx-img.cols+1, w, w );
            }
        }
    }
}

void VideoSnapCut::assignGMMsComponents( const Mat& img, const Mat& mask, const GMM& bgdGMM, const GMM& fgdGMM, Mat& compIdxs )
{
    Point p;
    for( p.y = 0; p.y < img.rows; p.y++ )
    {
        for( p.x = 0; p.x < img.cols; p.x++ )
        {
            Vec3d color = img.at<Vec3b>(p);
            compIdxs.at<int>(p) = mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD ?
                bgdGMM.whichComponent(color) : fgdGMM.whichComponent(color);
        }
    }
}

void VideoSnapCut::initGMMs( const Mat& img, const Mat& mask, GMM& bgdGMM, GMM& fgdGMM )
{
    const int kMeansItCount = 10;
    const int kMeansType = KMEANS_PP_CENTERS;

    Mat bgdLabels, fgdLabels;
    vector<Vec3f> bgdSamples, fgdSamples;
    Point p;
    for( p.y = 0; p.y < img.rows; p.y++ )
    {
        for( p.x = 0; p.x < img.cols; p.x++ )
        {
            if( mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD )
                bgdSamples.push_back( (Vec3f)img.at<Vec3b>(p) );
            else // GC_FGD | GC_PR_FGD
                fgdSamples.push_back( (Vec3f)img.at<Vec3b>(p) );
        }
    }
    CV_Assert( !bgdSamples.empty() && !fgdSamples.empty() );
    Mat _bgdSamples( (int)bgdSamples.size(), 3, CV_32FC1, &bgdSamples[0][0] );
    kmeans( _bgdSamples, GMM::componentsCount, bgdLabels,
            TermCriteria( CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType );
    Mat _fgdSamples( (int)fgdSamples.size(), 3, CV_32FC1, &fgdSamples[0][0] );
    kmeans( _fgdSamples, GMM::componentsCount, fgdLabels,
            TermCriteria( CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType );

    bgdGMM.initLearning();
    for( int i = 0; i < (int)bgdSamples.size(); i++ )
        bgdGMM.addSample( bgdLabels.at<int>(i,0), bgdSamples[i] );
    bgdGMM.endLearning();

    fgdGMM.initLearning();
    for( int i = 0; i < (int)fgdSamples.size(); i++ )
        fgdGMM.addSample( fgdLabels.at<int>(i,0), fgdSamples[i] );
    fgdGMM.endLearning();
}

void VideoSnapCut::learnGMMs( const Mat& img, const Mat& mask, const Mat& compIdxs, GMM& bgdGMM, GMM& fgdGMM )
{
    bgdGMM.initLearning();
    fgdGMM.initLearning();
    Point p;
    for( int ci = 0; ci < GMM::componentsCount; ci++ )
    {
        for( p.y = 0; p.y < img.rows; p.y++ )
        {
            for( p.x = 0; p.x < img.cols; p.x++ )
            {
                if( compIdxs.at<int>(p) == ci )
                {
                    if( mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD )
                        bgdGMM.addSample( ci, img.at<Vec3b>(p) );
                    else
                        fgdGMM.addSample( ci, img.at<Vec3b>(p) );
                }
            }
        }
    }
    bgdGMM.endLearning();
    fgdGMM.endLearning();
}

void VideoSnapCut::mGlobalGMM(const Mat& sourceImage, const Mat& binMask, const Mat& prof, Mat& fgdPro, Mat& bgdPro)
{
	////根据局部分类器的结果将图像分为4部分，0.2，0.5，0.8背景，可能背景，可能前景，前景
	Mat mask(sourceImage.size(), CV_8UC1);     //mask存储的是triMap
	Point p;
	int x1 = 0, x2 = 0, x3 = 0, x4 = 0;
	float minx = 1000000.0;
	float maxx = 0.0;
	for (p.y = 0; p.y < sourceImage.rows; p.y++) {
        for (p.x = 0; p.x < sourceImage.cols; p.x++) {
			if(minx > prof.at<float>(p))
				minx = prof.at<float>(p);
			if(maxx < prof.at<float>(p))
				maxx = prof.at<float>(p);
			if(prof.at<float>(p) < 0.000001)
			{
				x1++;
				mask.at<uchar>(p) = GC_BGD;
			}
			else if(prof.at<float>(p) >= 0.000001 && prof.at<float>(p) < 0.5)
			{
				x2++;
				mask.at<uchar>(p) = GC_PR_BGD;
			}
			else if(prof.at<float>(p) >= 0.5 && prof.at<float>(p) < 0.999999)
			{
				x3++;
				mask.at<uchar>(p) = GC_PR_FGD;
			}
			else
			{
				x4++;
				mask.at<uchar>(p) = GC_FGD;
			}
        }
    }
	cout << "x1 = " << x1 << ", x2 = " << x2 << ", x3 = " << x3 << ", x4 = " << x4 << ", minx = " << minx << ", maxx = " << maxx << endl;
	showTriMap(mask);
	Mat fgdModel, bgdModel;
	GMM fgdGMM(fgdModel);
	GMM bgdGMM(bgdModel);	
	initGMMs( sourceImage, binMask, bgdGMM, fgdGMM );
	Mat compIdxs( sourceImage.size(), CV_32SC1 );
	assignGMMsComponents( sourceImage, binMask, bgdGMM, fgdGMM, compIdxs );
    learnGMMs( sourceImage, binMask, compIdxs, bgdGMM, fgdGMM );
    mask.copyTo(fgdPro);
}

void VideoSnapCut::showTriMap(const Mat& triMap)
{
	Mat res(triMap.size(), CV_8UC1);
	Point p;
	for(p.y = 0; p.y < triMap.rows; p.y++){
		for(p.x = 0; p.x < triMap.cols; p.x++)
		{
			if(triMap.at<uchar>(p) == GC_BGD)
			{
				res.at<uchar>(p) = 0;
			}
			else if(triMap.at<uchar>(p) == GC_PR_BGD)
			{
				res.at<uchar>(p) = 80;
			}
			else if(triMap.at<uchar>(p) == GC_PR_FGD)
			{
				res.at<uchar>(p) = 165;
			}
			else if(triMap.at<uchar>(p) == GC_FGD)
			{
				res.at<uchar>(p) = 255;
			}
		}
	}
	imshow("由分类器概率生成的triMap", res);
}
//#include "instance.inc"