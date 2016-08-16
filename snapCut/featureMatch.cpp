//#include <math.h>
//#include <cv.h>
//#include <cxcore.h>
//#include <highgui.h>
//#include "DataStructures.h"
//extern "C"
//{
//#include "sift.h"
//#include "imgfeatures.h"
//#include "kdtree.h"
//#include "utils.h"
//#include "xform.h"
//}
//using namespace cv;
//int main1()
//{
//	Mat image1 = imread("E:\\test\\images\\0040.jpg");
//	Mat image2 = imread("E:\\test\\images\\0041.jpg");
//	IplImage img1 = image1;
//	IplImage img2 = image2;
//
//	struct feature* feat1, * feat2, * feat;//图像特征
//	struct feature** nbrs;
//	struct kd_node* kd_root;
//	CvPoint pt1, pt2;
//	double d0, d1;
//	int n1, n2, k, i, m = 0;
//
//	IplImage* stacked = stack_imgs( &img1, &img2 );//合并两张图像
//
//	n1 = sift_features( &img1, &feat1 );//获取shift特征点和特征点数
//
//	n2 = sift_features( &img2, &feat2 );
//	
//	//	select features inside the object
//	int n1In = 0;
//	for( i = 0; i < n1; i++ )
//	{
//		feat = feat1 + i;
//		int x = (int)feat->x;
//		int y = (int)feat->y;
//		/*if(mCombinedMask.at<uchar>(y, x) == 1)
//		{
//			feat1[n1In] = (*feat);
//			n1In++;
//		}*/
//	}
//
//	//fprintf( stderr, "Building kd tree...\n" );
//	kd_root = kdtree_build( feat2, n2 );
//	
//	for( i = 0; i < n1In; i++ )
//	{
//		feat = feat1 + i;
//		k = kdtree_bbf_knn( kd_root, feat, 2, &nbrs, KDTREE_BBF_MAX_NN_CHKS );
//		if( k == 2 ) 
//		{
//			d0 = descr_dist_sq( feat, nbrs[0] );
//			d1 = descr_dist_sq( feat, nbrs[1] );
//			if( d0 < d1 * NN_SQ_DIST_RATIO_THR )
//			{
//				pt1 = cvPoint( cvRound( feat->x ), cvRound( feat->y ) );
//				pt2 = cvPoint( cvRound( nbrs[0]->x ), cvRound( nbrs[0]->y ) );
//				pt2.y += img1.height;
//				cvLine( stacked, pt1, pt2, CV_RGB(255,0,255), 1, 8, 0 );
//				m++;
//				feat1[i].fwd_match = nbrs[0];
//			}
//		}
//		free( nbrs );
//	}
//	return 1;
//}