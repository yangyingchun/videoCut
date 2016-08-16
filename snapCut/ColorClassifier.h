#ifndef __COLORCLASSIFIER__H__ 
#define __COLORCLASSIFIER__H__ 

#include "GMM.h"

class ColorClassifier;
struct PixelClassifier
{
	ColorClassifier *mColorClassifier[4];
	float mClassifierP_c_x[4];		 // probability
	float mClassifierWx[4];		     // weight
	float mClassifierF_s_x[4];		// weight
	int mNumClassifiers;
	PixelClassifier()
	{
		mNumClassifiers = 0;         //��¼��λ���м���������
		
		mColorClassifier[0] = 0;
		mColorClassifier[1] = 0;
		mColorClassifier[2] = 0;
		mColorClassifier[3] = 0;


		mClassifierP_c_x[0] = 0;
		mClassifierP_c_x[1] = 0;
		mClassifierP_c_x[2] = 0;
		mClassifierP_c_x[3] = 0;

		mClassifierWx[0] = 0;
		mClassifierWx[1] = 0;
		mClassifierWx[2] = 0;
		mClassifierWx[3] = 0;

		mClassifierF_s_x[0] = 0;
		mClassifierF_s_x[1] = 0;
		mClassifierF_s_x[2] = 0;
		mClassifierF_s_x[3] = 0;
	}

};

class ColorClassifier
{
public:
	ColorClassifier(void);
		
	int Build();
	void FreeGMM();

	~ColorClassifier(void);
	int BuildBand();
public:
	Mat					mImageF;
	DataImage<PixelClassifier>*			mDataImagePixelClassifier;
	Mat					mCombinedMask;
	Mat					mDistanceTransform;
	float				mBoundingBoxCenter[2];

	GMM*				mForegroundGMM;
	GMM*				mBackgroundGMM;

	//խ��
	Mat                 mBand;
	GMM*                mForegroundGMM_band;
	GMM*                mBackgroundGMM_band;  //��խ��������ϸ�˹ģ��
	float				mF_C;
	float               sigamS;
};


#endif