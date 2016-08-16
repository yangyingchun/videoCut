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
	void generateContourWithRect(Mat& tmp);//�Զ�������������������װ��mForegroundBorderPolygons��

	void CreateMasks();
	void ClearMasks();
	Mat CreateMask(const PolygonF* polygon);//���������ߴ���һ��mask����
	Mat CreateContour(const Mat& Mask);
	Mat CreateDistanceTransforms(const Mat& Border);
	Mat CreatSampleContourImage(const Mat& image);

	void SampleContour();               //���յõ�������ı�ž���mForegroundSampledBorderPolygons
	void ClearBorderSamples();          //ɾ������mForegroundSampledBorderPolygons
	
	void RegisterCurFrameWithFrame(const Mat& iFrame);

	void SetForegroundBorderPolygons(vector<PolygonF*>* ForegroundBorderPolygons);

	int BuildColorClassifiers();
	void FreeColorClassifiers();

	int ClassifyPixels();
private:   //��ϸ�˹����
	void mGlobalGMM(const Mat& sourceImage, const Mat& binMask, const Mat& prof, Mat& fgdPro, Mat& bgdPro);//���ݾֲ������������Ĵ�
	void initGMMs( const Mat& img, const Mat& mask, GMM& bgdGMM, GMM& fgdGMM );
	void learnGMMs( const Mat& img, const Mat& mask, const Mat& compIdxs, GMM& bgdGMM, GMM& fgdGMM );
	void assignGMMsComponents( const Mat& img, const Mat& mask, const GMM& bgdGMM, const GMM& fgdGMM, Mat& compIdxs );	
	void showTriMap(const Mat& triMap);    //triMap��cv_8c1
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
	DataImage<PixelClassifier>*			mDataImagePixelClassifier;//��SetCurFrameMat�г�ʼ����
	Mat									mCombinedMask;
	vector<Mat>							mDistanceTransforms;
	int									mCurFrame;//��ʾ��ǰ֡�ǵڼ�֡

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

//֡��ͬ������Ϊ��������
private:            
	int             mFrameNum;
public:	
	int  getFrameNum(){return mFrameNum;}

//խ���ָ��
public:
	Mat mBand;      //�洢խ����Ϣ�������򸲸ǵĲ���
	Mat mfPro;
	Mat CreateContourBand(const Mat& Mask);
	Mat mGrabcutResult;

	//������ʾ
	void makecolorwheel(vector<Scalar> &colorwheel);
	void motionToColor(Mat flow, Mat &color);

	Mat binMask;
	Mat binMask1;
	
public:
	Mat nextContourMat;     //cv_8uc1,��ֵ���ǹؼ�֡ʱ����һ֡�ָ�������gt.Ѱ������
};

#endif