/*
 * GrabCut implementation source code Copyright(c) 2005-2006 Justin Talbot
 *
 * All Rights Reserved.
 * For educational use only; commercial use expressly forbidden.
 * NO WARRANTY, express or implied, for this software.
 */

#ifndef GMM_H
#define GMM_H

#include "Color.h"
#include "GaussianFitter.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>
using namespace std;
using namespace cv;

//#include "cv.h"


class GMM
{
public:

	// Initialize GMM with number of gaussians desired.
	GMM(unsigned int K);
	~GMM();

	unsigned int K() const { return mK; }

	// Returns the probability density of color c in this GMM
	Real p(Color c);

	// Returns the probability density of color c in just Gaussian k
	Real p(unsigned int i, Color c);

	int Build(double** data, uint nrows);

private:
	bool flag;              //标志是用局部函数（true）还是全局函数（false）
	unsigned int mK;		// number of gaussians
	GaussianPDF* mGaussians;	// an array of K gaussians

public:
	//全局的混合高斯模型
    static const int componentsCount = 5;

    GMM( Mat& _model );
    double operator()( const Vec3d color ) const;
    double operator()( int ci, const Vec3d color ) const;
    int whichComponent( const Vec3d color ) const;

    void initLearning();
    void addSample( int ci, const Vec3d color );
    void endLearning();
private:
    void calcInverseCovAndDeterm( int ci );
    Mat model;
    double* coefs;
    double* mean;
    double* cov;

    double inverseCovs[componentsCount][3][3];
    double covDeterms[componentsCount];

    double sums[componentsCount][3];
    double prods[componentsCount][3][3];
    int sampleCounts[componentsCount];
    int totalSampleCount;


	//friend void buildGMMs(GMM& backgroundGMM, GMM& foregroundGMM, Image<unsigned int>& components, const Image<Color>& image, const Image<SegmentationValue>& hardSegmentation);
	//friend void learnGMMs(GMM& backgroundGMM, GMM& foregroundGMM, Image<unsigned int>& components, const Image<Color>& image, const Image<SegmentationValue>& hardSegmentation);
};

//// Build the initial GMMs using the Orchard and Bouman color clustering algorithm
//void buildGMMs(GMM& backgroundGMM, GMM& foregroundGMM, Image<unsigned int>& components, const Image<Color>& image, const Image<SegmentationValue>& hardSegmentation);
//
//// Iteratively learn GMMs using GrabCut updating algorithm
//void learnGMMs(GMM& backgroundGMM, GMM& foregroundGMM, Image<unsigned int>& components, const Image<Color>& image, const Image<SegmentationValue>& hardSegmentation);




#endif //GMM_H