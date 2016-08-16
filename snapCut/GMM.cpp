/*
 * GrabCut implementation source code Copyright(c) 2005-2006 Justin Talbot
 *
 * All Rights Reserved.
 * For educational use only; commercial use expressly forbidden.
 * NO WARRANTY, express or implied, for this software.
 */
#include "GMM.h"
#include "cluster.h"

GMM::GMM(unsigned int K) : mK(K)
{
	flag = true;
	mGaussians = new GaussianPDF[mK];
}

GMM::~GMM()
{
	if (flag && mGaussians)
		delete [] mGaussians;
}

Real GMM::p(Color c)
{
	Real result = 0;

	if (mGaussians)
	{
		for (unsigned int i=0; i < mK; i++)
		{
			result += mGaussians[i].pi * p(i, c);
		}
	}

	return result;
}

Real GMM::p(unsigned int i, Color c)
{
	Real result = 0;

	if( mGaussians[i].pi > 0 )
	{
		if (mGaussians[i].determinant > 0)
		{
			Real r = c.r - mGaussians[i].mu.r;
			Real g = c.g - mGaussians[i].mu.g;
			Real b = c.b - mGaussians[i].mu.b;
			
			Real d = r * (r*mGaussians[i].inverse[0][0] + g*mGaussians[i].inverse[1][0] + b*mGaussians[i].inverse[2][0]) +
					g * (r*mGaussians[i].inverse[0][1] + g*mGaussians[i].inverse[1][1] + b*mGaussians[i].inverse[2][1]) +
					b * (r*mGaussians[i].inverse[0][2] + g*mGaussians[i].inverse[1][2] + b*mGaussians[i].inverse[2][2]);

			result = (Real)(1.0/(sqrt(mGaussians[i].determinant)) * exp(-0.5*d));
		}
	}

	return result;
}

//vector<Color> ColorData
//uint nrows = ColorData.size();
//double** data = (double**)malloc(nrows*sizeof(double*));
//for (i = 0; i < nrows; i++) data[i] = (double*)malloc(3*sizeof(double));
//	copy the data from the color array to a temp array 
//	and assin each sample a random cluster id
//for (j = 0; j < nrows; j++)
//{
//	data[j][0] = ColorData[j].r;
//	data[j][1] = ColorData[j].g;
//	data[j][2] = ColorData[j].b;			
//}
//for (i = 0; i < nrows; i++) free(data[i]);
//free(data);
	
int GMM::Build(double** data, uint nrows)
{
	uint i,j;
	int* clusterid = (int*)malloc(nrows*sizeof(int));
	
	//	run	k-means clustering
	const int ncols = 3;		
	const int nclusters = mK;
	const int transpose = 0;
	const char dist = 'e';
	const char method = 'a';
	int npass = 1;
	int ifound = 0;
	double error;
	double* weight = (double*)malloc(ncols*sizeof(double));
	int** mask = (int**)malloc(nrows*sizeof(int*));
	for (i = 0; i < nrows; i++)
	{
		mask[i] = (int*)malloc(ncols*sizeof(int));
		for (j = 0; j < ncols; j++) mask[i][j] = 1;
	}

	for (i = 0; i < ncols; i++) weight[i] = 1.0;
	
	kcluster(nclusters,nrows,ncols,data,mask,weight,transpose,npass,method,dist,clusterid, &error, &ifound);
	

	////	for debuging 
	//int* count;
	//count = (int*)malloc(nclusters*sizeof(int));
	//for (j = 0; j < nclusters; j++) count[j] = 0;
	//count[clusterid[j]]++;
	//free(count);

	//	build the GMM 
	GaussianFitter* gaussianFitter = new GaussianFitter[mK];

	//for (j = 0; j < ncols; j++)
	for (i = 0; i < nrows; i++) gaussianFitter[clusterid[i]].add(Color((float)data[i][0],(float)data[i][1],(float)data[i][2]));

	for (i = 0; i < mK; i++) gaussianFitter[i].finalize(mGaussians[i], nrows);

	delete [] gaussianFitter;

	for (i = 0; i < nrows; i++) free(mask[i]);
	
	free(mask);	

	free(weight);
	
	free(clusterid);
	
	return 1;
}

//全局混合高斯模型实现
GMM::GMM( Mat& _model )
{
	flag = false;
    const int modelSize = 3/*mean*/ + 9/*covariance*/ + 1/*component weight*/;
    if( _model.empty() )
    {
        _model.create( 1, modelSize*componentsCount, CV_64FC1 );
        _model.setTo(Scalar(0));
    }
    else if( (_model.type() != CV_64FC1) || (_model.rows != 1) || (_model.cols != modelSize*componentsCount) )
        CV_Error( CV_StsBadArg, "_model must have CV_64FC1 type, rows == 1 and cols == 13*componentsCount" );

    model = _model;

    coefs = model.ptr<double>(0);
    mean = coefs + componentsCount;
    cov = mean + 3*componentsCount;

    for( int ci = 0; ci < componentsCount; ci++ )
        if( coefs[ci] > 0 )
             calcInverseCovAndDeterm( ci );
}

double GMM::operator()( const Vec3d color ) const
{
    double res = 0;
    for( int ci = 0; ci < componentsCount; ci++ )
        res += coefs[ci] * (*this)(ci, color );
    return res;
}

double GMM::operator()( int ci, const Vec3d color ) const
{
    double res = 0;
    if( coefs[ci] > 0 )
    {
        CV_Assert( covDeterms[ci] > std::numeric_limits<double>::epsilon() );
        Vec3d diff = color;
        double* m = mean + 3*ci;
        diff[0] -= m[0]; diff[1] -= m[1]; diff[2] -= m[2];
        double mult = diff[0]*(diff[0]*inverseCovs[ci][0][0] + diff[1]*inverseCovs[ci][1][0] + diff[2]*inverseCovs[ci][2][0])
                   + diff[1]*(diff[0]*inverseCovs[ci][0][1] + diff[1]*inverseCovs[ci][1][1] + diff[2]*inverseCovs[ci][2][1])
                   + diff[2]*(diff[0]*inverseCovs[ci][0][2] + diff[1]*inverseCovs[ci][1][2] + diff[2]*inverseCovs[ci][2][2]);
        res = 1.0f/sqrt(covDeterms[ci]) * exp(-0.5f*mult);
    }
    return res;
}

int GMM::whichComponent( const Vec3d color ) const
{
    int k = 0;
    double max = 0;

    for( int ci = 0; ci < componentsCount; ci++ )
    {
        double p = (*this)( ci, color );
        if( p > max )
        {
            k = ci;
            max = p;
        }
    }
    return k;
}

void GMM::initLearning()
{
    for( int ci = 0; ci < componentsCount; ci++)
    {
        sums[ci][0] = sums[ci][1] = sums[ci][2] = 0;
        prods[ci][0][0] = prods[ci][0][1] = prods[ci][0][2] = 0;
        prods[ci][1][0] = prods[ci][1][1] = prods[ci][1][2] = 0;
        prods[ci][2][0] = prods[ci][2][1] = prods[ci][2][2] = 0;
        sampleCounts[ci] = 0;
    }
    totalSampleCount = 0;
}

void GMM::addSample( int ci, const Vec3d color )
{
    sums[ci][0] += color[0]; sums[ci][1] += color[1]; sums[ci][2] += color[2];
    prods[ci][0][0] += color[0]*color[0]; prods[ci][0][1] += color[0]*color[1]; prods[ci][0][2] += color[0]*color[2];
    prods[ci][1][0] += color[1]*color[0]; prods[ci][1][1] += color[1]*color[1]; prods[ci][1][2] += color[1]*color[2];
    prods[ci][2][0] += color[2]*color[0]; prods[ci][2][1] += color[2]*color[1]; prods[ci][2][2] += color[2]*color[2];
    sampleCounts[ci]++;
    totalSampleCount++;
}

void GMM::endLearning()
{
    const double variance = 0.01;
    for( int ci = 0; ci < componentsCount; ci++ )
    {
        int n = sampleCounts[ci];
        if( n == 0 )
            coefs[ci] = 0;
        else
        {
            coefs[ci] = (double)n/totalSampleCount;

            double* m = mean + 3*ci;
            m[0] = sums[ci][0]/n; m[1] = sums[ci][1]/n; m[2] = sums[ci][2]/n;

            double* c = cov + 9*ci;
            c[0] = prods[ci][0][0]/n - m[0]*m[0]; c[1] = prods[ci][0][1]/n - m[0]*m[1]; c[2] = prods[ci][0][2]/n - m[0]*m[2];
            c[3] = prods[ci][1][0]/n - m[1]*m[0]; c[4] = prods[ci][1][1]/n - m[1]*m[1]; c[5] = prods[ci][1][2]/n - m[1]*m[2];
            c[6] = prods[ci][2][0]/n - m[2]*m[0]; c[7] = prods[ci][2][1]/n - m[2]*m[1]; c[8] = prods[ci][2][2]/n - m[2]*m[2];

            double dtrm = c[0]*(c[4]*c[8]-c[5]*c[7]) - c[1]*(c[3]*c[8]-c[5]*c[6]) + c[2]*(c[3]*c[7]-c[4]*c[6]);
            if( dtrm <= std::numeric_limits<double>::epsilon() )
            {
                // Adds the white noise to avoid singular covariance matrix.
                c[0] += variance;
                c[4] += variance;
                c[8] += variance;
            }

            calcInverseCovAndDeterm(ci);
        }
    }
}

void GMM::calcInverseCovAndDeterm( int ci )
{
    if( coefs[ci] > 0 )
    {
        double *c = cov + 9*ci;
        double dtrm =
              covDeterms[ci] = c[0]*(c[4]*c[8]-c[5]*c[7]) - c[1]*(c[3]*c[8]-c[5]*c[6]) + c[2]*(c[3]*c[7]-c[4]*c[6]);

        CV_Assert( dtrm > std::numeric_limits<double>::epsilon() );
        inverseCovs[ci][0][0] =  (c[4]*c[8] - c[5]*c[7]) / dtrm;
        inverseCovs[ci][1][0] = -(c[3]*c[8] - c[5]*c[6]) / dtrm;
        inverseCovs[ci][2][0] =  (c[3]*c[7] - c[4]*c[6]) / dtrm;
        inverseCovs[ci][0][1] = -(c[1]*c[8] - c[2]*c[7]) / dtrm;
        inverseCovs[ci][1][1] =  (c[0]*c[8] - c[2]*c[6]) / dtrm;
        inverseCovs[ci][2][1] = -(c[0]*c[7] - c[1]*c[6]) / dtrm;
        inverseCovs[ci][0][2] =  (c[1]*c[5] - c[2]*c[4]) / dtrm;
        inverseCovs[ci][1][2] = -(c[0]*c[5] - c[2]*c[3]) / dtrm;
        inverseCovs[ci][2][2] =  (c[0]*c[4] - c[1]*c[3]) / dtrm;
    }
}