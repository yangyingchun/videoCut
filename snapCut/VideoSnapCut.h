#ifndef __VIDEOSNAPCUT__H__ 
#define __VIDEOSNAPCUT__H__ 

#include "DataImage.h"
#include "Color.h"
#include "DataStructures.h"
#include "ColorClassifier.h"
#include "DataImage.h"
#include "VideoReader.h"
#include "graph.h"

typedef Graph<double, double, double> GraphType;

class VideoSnapCut
{
public:
	VideoSnapCut(void);
	void setFrameNum(int count){mFrameNum = count;}
	void generateContourWithRect(Mat& tmp);//自动生成轮廓，将采样点装入mForegroundBorderPolygons中

	void CreateMasks();
	void ClearMasks();
	Mat CreateMask(const PolygonF* polygon);//根据轮廓线创建一个mask矩阵
	Mat CreateContour(const Mat& Mask);
	Mat CreateDistanceTransforms(const Mat& Border);
	Mat CreatSampleContourImage(const Mat& image);

	void SampleContour();               //最终得到采样点的编号矩阵mForegroundSampledBorderPolygons
	void ClearBorderSamples();          //删除所有mForegroundSampledBorderPolygons
	
	void RegisterCurFrameWithFrame(const Mat& iFrame);

	void SetForegroundBorderPolygons(vector<PolygonF*>* ForegroundBorderPolygons);

	int BuildColorClassifiers();
	void FreeColorClassifiers();

	int ClassifyPixels();
private:   //混合高斯处理
	void mGlobalGMM(const Mat& sourceImage, const Mat& binMask, const Mat& prof, Mat& fgdPro, Mat& bgdPro);//根据局部分类器建立的带
	void initGMMs( const Mat& img, const Mat& mask, GMM& bgdGMM, GMM& fgdGMM );
	void learnGMMs( const Mat& img, const Mat& mask, const Mat& compIdxs, GMM& bgdGMM, GMM& fgdGMM );
	void assignGMMsComponents( const Mat& img, const Mat& mask, const GMM& bgdGMM, const GMM& fgdGMM, Mat& compIdxs );	
	void showTriMap(const Mat& triMap);    //triMap是cv_8c1
	void BandCut(const Mat& img, Mat& mask, Mat& fgdPro);
	double calcBeta(const Mat& img);
	void calcNWeights(const Mat& img, Mat& leftW, Mat& upleftW, Mat& upW,
						    Mat& uprightW, double beta, double gamma);
	void constructGCGraph( const Mat& img, const Mat& mask, const GMM& bgdGMM, const GMM& fgdGMM, double lambda,
                           const Mat& leftW, const Mat& upleftW, const Mat& upW, const Mat& uprightW,
                           GraphType* graph );
	void BandCut1(const Mat& img, Mat& mask, Mat& fgdPro);
	void constructGCGraph(const Mat& img, const Mat& imgPro, double lambda, const Mat& leftW, const Mat& upleftW,
                          const Mat& upW, const Mat& uprightW, GraphType* graph);
	void estimateSegmentation(GraphType* graph, Mat& mask);

public:
	void SetCurFrameMat(Mat CurFrameMat);
	void SetCurFrameMatOnly(Mat CurFrameMat);
	void FreeDataImagePixelClassifier();

	void SetCurFrame(int CurFrame);


	~VideoSnapCut(void);
	bool PointInPolygon(const PolygonF* polygon, float y, float x) ;
	void DisplayImage(char* name, const Mat& mat, int channels = 1, bool multiply = true);
	void DisplayDistanceImage(char* name, const Mat& mat, int channels = 1, bool multiply = true);		
	void DisplayBorderSamples(int i);
	void DisplayBorderSamples();
	void ResampleContour();
	
public:	
	Mat									mImage;
	Mat									mImageF;
	DataImage<PixelClassifier>*			mDataImagePixelClassifier;//在SetCurFrameMat中初始化的
	Mat									mCombinedMask;
	vector<Mat>							mDistanceTransforms;
	int									mCurFrame;//显示当前帧是第几帧

	vector<PolygonF*>					mForegroundSampledBorderPolygons;	
	vector<PolygonF*>*					mForegroundBorderPolygons;
	vector<vector<ColorClassifier*>>	mColorClassifiers;

	Mat                                 mPreImage;
	Mat                                 mPreImageF;

	vector<Mat>							mMasks;

	vector<Mat>							mBorders;

	vector<Mat>							mSampledContourImages;
	vector<Mat>                         mAfterTransContours;
	
public:
	static int		mMaskSize;
	static int		mMaskSize2; // /2
	static float	mSigmaS;
	static float	mSigmaS2;
	static int		mK;
	static int		mMaxColorClassifiers;

//帧数同步，作为输出结果用
private:            
	int             mFrameNum;
public:	
	int  getFrameNum(){return mFrameNum;}

//窄带分割部分
public:
	Mat mBand;      //存储窄带信息，即方框覆盖的部分
	Mat mfPro;
	Mat CreateContourBand(const Mat& Mask);
	Mat mGrabcutResult;

	//光流显示
	void makecolorwheel(vector<Scalar> &colorwheel);
	void motionToColor(Mat flow, Mat &color);

	Mat binMask;
	Mat binMask1;
	
public:
	Mat nextContourMat;     //cv_8uc1,二值，非关键帧时将上一帧分割结果传给gt.寻找轮廓
};

#endif