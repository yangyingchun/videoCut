#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include <math.h>
#include <iostream>
#include "VideoSnapCut.h"
#include <fstream>
#include "constSet.h"

#define UNKNOWN_FLOW_THRESH 1e9

using namespace std;
using namespace cv;


class GCApplication
{
public:
	GCApplication();
    void setImageAndWinName( const Mat& _image, const string& _winName );
    void showJiaoHuImage();
    void reset();
    void mouseClick( int event, int x, int y, int flags, void* param );
    void setRectInMask();
    void setLblsInMask( int flags, Point p, bool isPr );
    int getIterCount() const { return iterCount; }
    int nextIter();
	void saveCutResult();
	void transMatTocontours(const Mat& conMat, vector<vector<Point>>& vP);
	PolygonF* showContour(vector<Point>& vp, Mat& res, float dist, vector<Rect>& vr, int rat = 5);


public://预处理工具
	void getBinMask( const Mat& comMask, Mat& binMask );
	void makecolorwheel(vector<Scalar> &colorwheel);
	void motionToColor(Mat flow, Mat &color);
public:
    enum{ NOT_SET = 0, IN_PROCESS = 1, SET = 2 };
    static const int radius = 2;
    static const int thickness = -1;


	void showImageD();
	void showImageDwithRect();

	void showImage();
	void setRect(int event, int x, int y, int flags, void* param);

	int nextFrameIter();


	Mat resMaskN;

	vector<Rect> rectDs;//存储边缘跟踪框
	vector<Rect> rectDsPos;//通过光流变换后的矩形框
	Mat imageNext;
	void setRectsInMask();//将第二帧得到的矩形框覆盖的区域设为可能前景区域
	Mat maskN;//第二帧的可能前景
public:




	Mat bgdModelN, fgdModelN;

    
	uchar rectDState;
    


	Rect rectD;//跟踪框
	
	vector<Point> fgdPxlsN, bgdPxlsN, prFgdPxlsN, prBgdPxlsN;

	int iterCountN;
public:
	Mat cannyContour;
public:
	VideoSnapCut* mVideoSnapCut;
	void snapcut();
	vector<PolygonF*> ForegroundBorderPolygons;
	Mat getKFrame(int k, VideoCapture& cap);  //得到第k帧图像
	void interactiveTool(const string& winName);
	void dilateAndErode(Mat& con, int n);
	vector<PolygonF*> normalPolygons(vector<PolygonF*>& fgdPolygons, float dist);
	PolygonF* normalPolygon(PolygonF* fgdPolygon, float dist);
	void showPolygonF(PolygonF* pf, Mat& res, float dist, vector<Rect>& vr);
	
	vector<vector<Point>>& generateContours(const Mat& con);
	float getDistanceOfPoint(Point p1, Point p2);
	Mat getCanny(Mat img);
	vector<Rect> changeRects(vector<Rect> v1, vector<Rect> v2, const Mat& cannyContour);
	PolygonF* pointToF(vector<Point>& p);
	void showRects(Mat& res, vector<Rect>& vr);    //在图像上显示采样框
	void updateRects(vector<Rect>& vrs, vector<Rect>& vrt);

public:
	Mat resMask;
	Mat resbinMask;//记录两种形式的分割结果

	const string* winName;
    const Mat* image;
	Mat mask;

	vector<Point> fgdPxls, bgdPxls, prFgdPxls, prBgdPxls;
	bool isInitialized;
	uchar rectState, lblsState, prLblsState;
    int iterCount;

	Rect rect;

	Mat bgdModel, fgdModel;//
};

void GCApplication::setImageAndWinName( const Mat& _image, const string& _winName  )
{
    if( _image.empty() || _winName.empty() )
        return;
    image = &_image;
    winName = &_winName;
    mask.create( image->size(), CV_8UC1);
    reset();
}

void GCApplication::reset()
{
    if( !mask.empty() )
        mask.setTo(Scalar::all(GC_BGD));
    bgdPxls.clear(); fgdPxls.clear();
    prBgdPxls.clear();  prFgdPxls.clear();//前景和背景标记像素容器清零，三类标记清零，迭代次数清

    isInitialized = false;
    rectState = NOT_SET;
    lblsState = NOT_SET;
    prLblsState = NOT_SET;
    iterCount = 0;
}

void GCApplication::mouseClick( int event, int x, int y, int flags, void* )
{
    // TODO add bad args check
    switch( event )
    {
    case CV_EVENT_LBUTTONDOWN: // set rect or GC_BGD(GC_FGD) labels
        {
            bool isb = (flags & BGD_KEY) != 0,
                 isf = (flags & FGD_KEY) != 0;
            if( rectState == NOT_SET && !isb && !isf )
            {
                rectState = IN_PROCESS;
                rect = Rect( x, y, 1, 1 );
            }
            if ( (isb || isf) && rectState == SET )
                lblsState = IN_PROCESS;
        }
        break;
    case CV_EVENT_RBUTTONDOWN: // set GC_PR_BGD(GC_PR_FGD) labels
        {
            bool isb = (flags & BGD_KEY) != 0,
                 isf = (flags & FGD_KEY) != 0;
            if ( (isb || isf) && rectState == SET )
                prLblsState = IN_PROCESS;
        }
        break;
    case CV_EVENT_LBUTTONUP:
        if( rectState == IN_PROCESS )
        {
            rect = Rect( Point(rect.x, rect.y), Point(x,y) );
            rectState = SET;
            setRectInMask();
            assert( bgdPxls.empty() && fgdPxls.empty() && prBgdPxls.empty() && prFgdPxls.empty() );
			ofstream saveJiaohu("E:\\shiyanResult\\jiaohuRect.txt");
			saveJiaohu << "rect.x = " << rect.x << ", rect.y = " << rect.y << ", rect.width = " << rect.width << ", rect.height = " << rect.height;
			saveJiaohu.close();
            showImage();
        }
        if( lblsState == IN_PROCESS )
        {
            setLblsInMask(flags, Point(x,y), false);
            lblsState = SET;
            showImage();
        }
        break;
    case CV_EVENT_RBUTTONUP:
        if( prLblsState == IN_PROCESS )
        {
            setLblsInMask(flags, Point(x,y), true);
            prLblsState = SET;
            showImage();
        }
        break;
    case CV_EVENT_MOUSEMOVE:
        if( rectState == IN_PROCESS )
        {
            rect = Rect( Point(rect.x, rect.y), Point(x,y) );
            assert( bgdPxls.empty() && fgdPxls.empty() && prBgdPxls.empty() && prFgdPxls.empty() );
            showImage();
        }
        else if( lblsState == IN_PROCESS )
        {
            setLblsInMask(flags, Point(x,y), false);
            showImage();
        }
        else if( prLblsState == IN_PROCESS )
        {
            setLblsInMask(flags, Point(x,y), true);
            showImage();
        }
        break;
    }
}

void GCApplication::showJiaoHuImage()
{
    if( image->empty() || winName->empty() )
        return;

    Mat res;
	image->copyTo(res);
 
    vector<Point>::const_iterator it;
    for( it = bgdPxls.begin(); it != bgdPxls.end(); ++it )
        circle( res, *it, radius, BLUE, thickness );
    for( it = fgdPxls.begin(); it != fgdPxls.end(); ++it )
        circle( res, *it, radius, RED, thickness );
    for( it = prBgdPxls.begin(); it != prBgdPxls.end(); ++it )
        circle( res, *it, radius, LIGHTBLUE, thickness );
    for( it = prFgdPxls.begin(); it != prFgdPxls.end(); ++it )
        circle( res, *it, radius, PINK, thickness );

    if( rectState == IN_PROCESS || rectState == SET)
        rectangle( res, Point( rect.x, rect.y ), Point(rect.x + rect.width, rect.y + rect.height ), GREEN, 2);
	
    imshow( *winName, res );
}

void GCApplication::setRectInMask()
{
    assert( !mask.empty() );
    mask.setTo( GC_BGD );
    rect.x = max(0, rect.x);
    rect.y = max(0, rect.y);
    rect.width = min(rect.width, image->cols-rect.x);
    rect.height = min(rect.height, image->rows-rect.y);
    (mask(rect)).setTo( Scalar(GC_PR_FGD) );//在mat中设置矩形框的值为可能前景
}

void GCApplication::setLblsInMask( int flags, Point p, bool isPr )
{
    vector<Point> *bpxls, *fpxls;
    uchar bvalue, fvalue;
    if( !isPr )
    {
        bpxls = &bgdPxls;
        fpxls = &fgdPxls;
        bvalue = GC_BGD;
        fvalue = GC_FGD;
    }
    else
    {
        bpxls = &prBgdPxls;
        fpxls = &prFgdPxls;
        bvalue = GC_PR_BGD;
        fvalue = GC_PR_FGD;
    }
    if( flags & BGD_KEY )
    {
        bpxls->push_back(p);
        circle( mask, p, radius, bvalue, thickness );
    }
    if( flags & FGD_KEY )
    {
        fpxls->push_back(p);
        circle( mask, p, radius, fvalue, thickness );
    }
}

int GCApplication::nextIter()
{
    if( isInitialized ) //非第一次分割
        grabCut( *image, mask, rect, bgdModel, fgdModel, 1 );
    else                //第一次分割
    {
        if( rectState != SET )
            return iterCount;

        if( lblsState == SET || prLblsState == SET )//根据交互的前景背景种子点进行grabcut分割
            grabCut( *image, mask, rect, bgdModel, fgdModel, 1, GC_INIT_WITH_MASK );
        else
            grabCut( *image, mask, rect, bgdModel, fgdModel, 1, GC_INIT_WITH_RECT );

        isInitialized = true;
    }
    iterCount++;

    bgdPxls.clear(); fgdPxls.clear();              //分割完清除种子点
    prBgdPxls.clear(); prFgdPxls.clear();

    return iterCount;
}

void GCApplication::saveCutResult()
{
	if( image->empty() || winName->empty() )
        return;

    Mat res;
    Mat binMask;
    if( !isInitialized )
        image->copyTo( res );
    else
    {
        getBinMask( mask, binMask );
        image->copyTo( res, binMask );
    }
	binMask.copyTo(resbinMask);
	res.copyTo(resMask);
}
void GCApplication::showImage()
{
    if( image->empty() || winName->empty() )
        return;

    Mat res;
    Mat binMask;
    if( !isInitialized )
        image->copyTo( res );
    else
    {
        getBinMask( mask, binMask );
        image->copyTo( res, binMask );
    }
	binMask.copyTo(resbinMask);
	res.copyTo(resMask);
 
    vector<Point>::const_iterator it;
    for( it = bgdPxls.begin(); it != bgdPxls.end(); ++it )
        circle( res, *it, radius, BLUE, thickness );
    for( it = fgdPxls.begin(); it != fgdPxls.end(); ++it )
        circle( res, *it, radius, RED, thickness );
    for( it = prBgdPxls.begin(); it != prBgdPxls.end(); ++it )
        circle( res, *it, radius, LIGHTBLUE, thickness );
    for( it = prFgdPxls.begin(); it != prFgdPxls.end(); ++it )
        circle( res, *it, radius, PINK, thickness );

    if( rectState == IN_PROCESS || rectState == SET)
        rectangle( res, Point( rect.x, rect.y ), Point(rect.x + rect.width, rect.y + rect.height ), GREEN, 2);
	
    imshow( *winName, res );
}

void GCApplication::transMatTocontours(const Mat& conMat, vector<vector<Point>>& vP)
{
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(conMat, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);				
	vP = contours;
}

void GCApplication::updateRects(vector<Rect>& vrs, vector<Rect>& vrt)
{
	vrt.clear();
	int n = vrs.size();
	PolygonF* pf = new PolygonF();
	for(int i = 0; i < n - 1; i++)
	{
		int centerX = vrs[i].x + vrs[i].width/2;
		int centerY = vrs[i].y + vrs[i].height/2;
		PointF p(centerX, centerY);
		pf->mPoints.push_back(p);
	}
	float dist = vrs[0].width*2.0/3.0;
	//vector<PolygonF*> pfv;
	//pfv.push_back(pf);
	PolygonF* pfNew = normalPolygon(pf, dist);
	for(int i = 0; i < pfNew->mPoints.size(); i++)
	{
		PointF pF = pfNew->mPoints[i];
		int tmpX = pF.x;
		int width = vrs[0].width;
		int tmpY = pF.y;
		int height = vrs[0].height;
		Point p1(tmpX, tmpY);
		Point p2(tmpX + width, tmpY + height);
		vrt.push_back(Rect(p1, p2));
	}
	vrt.push_back(vrs[n-1]);
}

vector<Rect> GCApplication::changeRects(vector<Rect> v1, vector<Rect> v2, const Mat& cannyContour)
{
	int n = v1.size();
	assert(n >= 3);//轮廓点小于3时抛出异常
	for(int i = 0; i%n < n; i++)
	{
		if(i == 3*n) break;
		int width = v1[i%n].width/2;
		int height = v2[i%n].height/2;
		Point centrePre(v1[i%n].x + width, v1[i%n].y + height);
		Point centreCur(v1[(i+1)%n].x + width, v1[(i+1)%n].y + height);
		Point centrePost(v1[(i+2)%n].x + width, v1[(i+2)%n].y + height);
		float dis1 = getDistanceOfPoint(centrePre, centreCur);
		float dis2 = getDistanceOfPoint(centreCur, centrePost);
		float thresh = 1.0;
		Point centreNew;
		while(dis1 - dis2 > thresh || dis2 - dis1 > thresh)
		{
			bool flag = false;
			for(int j = 1; j < width; j++)
			{
				for(int k = 1; k < height; k++)
				{
					if(cannyContour.at<uchar>(centreCur.x + j, centreCur.y + k) != 0)
					{
						centreNew.x = centreCur.x + j;
						centreNew.y = centreCur.y + k;
						dis1 = getDistanceOfPoint(centrePre, centreNew);
						dis2 = getDistanceOfPoint(centreNew, centrePost);
						flag = true;
					}
					if(flag == true)
						break;
				}
				if(flag == true)
					break;
			}
		}
		centreCur.x = centreNew.x;
		centreCur.y = centreNew.y;
	}
	return v1;
}

Mat GCApplication::getCanny(Mat img)
{
	Mat imageE, gray, edge, cedge;
	int edgeThresh = 100;
	img.copyTo(imageE);
	cedge.create(imageE.size(), imageE.type());
	cvtColor(imageE, gray, COLOR_BGR2GRAY);
	blur(gray, edge, Size(3,3));
	Canny(edge, edge, edgeThresh, edgeThresh*3, 3);
	cedge = Scalar::all(0);
	cannyContour = edge;
	imageE.copyTo(cedge, edge);
	imshow("Edge map", edge);
	return edge;
}

float GCApplication::getDistanceOfPoint(Point p1, Point p2)
{
	float res = 0.0;
	res = sqrt((float)(p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y));
	return res;
}
vector<vector<Point>>& GCApplication::generateContours(const Mat& con)
{
	vector<vector<Point>> result;
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	vector<Point> poly;
	findContours(con, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	for(int idx = 0; idx >= 0; idx = hierarchy[idx][0])//= hierarchy[idx][0]
	{	
		poly.clear();
		approxPolyDP(Mat(contours[idx]), poly, 0.001, true);
		result.push_back(poly);
	}	
	return result;
}

PolygonF* GCApplication::pointToF(vector<Point>& p)
{
	PolygonF* result = new PolygonF;
	vector<Point>::const_iterator itp = p.begin();
	while(itp != p.end())
	{
		PointF pf((float)((*itp).x), (float)((*itp).y));
		result->mPoints.push_back(pf);
		itp++;
	}
	return result;
}
PolygonF* GCApplication::showContour(vector<Point>& vp, Mat& res, float dist, vector<Rect>& vr, int rat)
{
	//rat表示由findcontour检测的轮空点向采样轮廓点过度的减小比
	////将这些点对应在图像res中的轮廓线画出来
	vector<Point>::const_iterator itp = vp.begin(); 
	while (itp != (vp.end() - 1)) 
	{ 
		line(res, *itp, *(itp + 1), Scalar(255), 1); 
		++itp; 
	} 
	line(res, *itp, *(vp.begin()), Scalar(255), 1);
	////将point向量转换成pointF*
	PolygonF* pyF = new PolygonF();
	for(int i = 0; i < vp.size(); i = i + rat)//缩减采样点比例
	{
		PointF pt;
		pt.x = vp[i].x;
		pt.y = vp[i].y;
		pyF->mPoints.push_back(pt);
	}
	PolygonF* pf = normalPolygon(pyF, dist);
	showPolygonF(pf, res, dist, vr);           
	return pyF;                             //返回的仍然是原始轮廓中的点以及轮廓上的均匀采样框
}

void GCApplication::showRects(Mat& res, vector<Rect>& vr)
{
	int n = vr.size();
	Point leftup, rightdown;
	leftup.x = res.cols - 1;
	leftup.y = res.rows - 1;
	rightdown.x = 0;
	rightdown.y = 0;
	for(int i = 0; i < n - 1; i++)
	{
		Point lu, rd;		
		lu.x = vr[i].x;
		lu.y = vr[i].y;
		rd.x = vr[i].x + vr[i].width;
		rd.y = vr[i].y + vr[i].height;
		rectangle( res, lu, rd, YELLOW, 1);	
		leftup.x = min(leftup.x, lu.x);
		leftup.y = min(leftup.y, lu.y);
		rightdown.x = max(rightdown.x, rd.x);
		rightdown.y = max(rightdown.y, rd.y);
	}
	leftup.x = max(leftup.x - 3, 0);
	leftup.y = max(leftup.y - 3, 0);
	rightdown.x = min(rightdown.x + 3, res.cols - 1);
	rightdown.y = min(rightdown.y + 3, res.rows - 1);
	vr[n - 1] = (Rect(leftup, rightdown));
	rectangle(res, leftup, rightdown, RED, 1);
}

void GCApplication::showPolygonF(PolygonF* pf, Mat& res, float dist, vector<Rect>& vr)//通过轮廓得到采样框
{
  /*vector<PointF>::iterator it = pf -> mPoints.begin();
	while (it != (pf -> mPoints.end() - 1)) //不用显示此时的多边形轮廓
	{ 
		line(res, Point((*it).x,(*it).y), Point((*(it + 1)).x,(*(it + 1)).y), Scalar(255), 1); 
		++it; 
	} 
	line(res, Point((*it).x,(*it).y), Point((*(pf -> mPoints.begin())).x,(*(pf -> mPoints.begin())).y), Scalar(255), 1);*/

	int dis = dist;
	vector<PointF> poly = pf -> mPoints;
	vr.clear();
	Point leftup, rightdown;
	leftup.x = res.cols - 1;
	leftup.y = res.rows - 1;
	rightdown.x = 0;
	rightdown.y = 0;
	for(int i = 0; i < pf -> mPoints.size(); i = i + 1)
	{
		Point lu, rd;
		lu.x = max(poly[i].x - dis, 0);
		lu.y = max(poly[i].y - dis, 0);
		rd.x = min(poly[i].x + dis, res.cols - 1);
		rd.y = min(poly[i].y + dis, res.rows - 1);
		Rect r(lu, rd);
		vr.push_back(r);
		rectangle( res, lu, rd, YELLOW, 1);	
		leftup.x = min(leftup.x, lu.x);
		leftup.y = min(leftup.y, lu.y);
		rightdown.x = max(rightdown.x, rd.x);
		rightdown.y = max(rightdown.y, rd.y);
	}
	//最后一个整个物体的跟踪框
	leftup.x = max(leftup.x - 3, 0);
	leftup.y = max(leftup.y - 3, 0);
	rightdown.x = min(rightdown.x + 3, res.cols - 1);
	rightdown.y = min(rightdown.y + 3, res.rows - 1);
	vr.push_back(Rect(leftup, rightdown));
	rectangle(res, leftup, rightdown, RED, 1);
}

PolygonF* GCApplication::normalPolygon(PolygonF* fgdPolygon, float dist)   //均衡化采样框
{
	int polygonSize = fgdPolygon -> mPoints.size();
	if(polygonSize < 1) return NULL;
	PolygonF* polygon = fgdPolygon;
	PolygonF* sampledPolygon = new PolygonF();
	float curDist = 0;
	int curPolygonPointIndex = 1;
	float curMagnitude;
	float sampleDist = dist;
	sampledPolygon -> mPoints.push_back(polygon -> mPoints[0]);
	PointF prevPoint = polygon -> mPoints[0];
	PointF curPoint;

	while(curPolygonPointIndex <= polygonSize)
	{
		int curPolygonPointIndexMod = curPolygonPointIndex % polygonSize;
		curPoint = polygon->mPoints[curPolygonPointIndexMod];
		curMagnitude = sqrt((curPoint.x - prevPoint.x)*(curPoint.x - prevPoint.x) + (curPoint.y - prevPoint.y)*(curPoint.y - prevPoint.y));
		if(curDist + curMagnitude < sampleDist)
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
			PointF newPoint = PointF(x0 + t*dx, y0 + t*dy);
			sampledPolygon -> mPoints.push_back(newPoint);
			prevPoint = newPoint;
			curDist = 0;
		}
	}
	PointF firstPoint = sampledPolygon -> mPoints[0];
	PointF lastPoint = sampledPolygon -> mPoints[sampledPolygon -> mPoints.size() - 1];
	float dx = lastPoint.x - firstPoint.x;
	float dy = lastPoint.y - firstPoint.y;
	float dis = sqrt(dx*dx + dy*dy);
	if(dis < sampleDist*1.0)
	{
		sampledPolygon -> mPoints.pop_back();
	}
	return sampledPolygon;
}

vector<PolygonF*> GCApplication::normalPolygons(vector<PolygonF*>& fgdPolygons, float dist)//将采样轮廓中的采样点均匀化
{
	vector<PolygonF*> result;
	for(uint i = 0; i < fgdPolygons.size(); i++)
	{
		//int polygonSize = fgdPolygons[i] -> mPoints.size();
		//if(polygonSize < 1) continue;
		//PolygonF* polygon = fgdPolygons[i];
		//PolygonF* sampledPolygon = new PolygonF();
		//float curDist = 0;
		//int curPolygonPointIndex = 1;
		//float curMagnitude;
		//float sampleDist = dist;
		//sampledPolygon -> mPoints.push_back(polygon -> mPoints[0]);
		//PointF prevPoint = polygon -> mPoints[0];
		//PointF curPoint;

		//while(curPolygonPointIndex <= polygonSize)
		//{
		//	int curPolygonPointIndexMod = curPolygonPointIndex % polygonSize;
		//	curPoint = polygon->mPoints[curPolygonPointIndexMod];
		//	curMagnitude = sqrt((curPoint.x - prevPoint.x)*(curPoint.x - prevPoint.x) + (curPoint.y - prevPoint.y)*(curPoint.y - prevPoint.y));
		//	if(curDist + curMagnitude < sampleDist)
		//	{
		//		curDist += curMagnitude;
		//		prevPoint = curPoint;
		//		curPolygonPointIndex++;
		//	}
		//	else
		//	{
		//		float remainingDist = sampleDist - curDist;
		//		float t = remainingDist / curMagnitude;
		//		float x0 = prevPoint.x;
		//		float y0 = prevPoint.y;
		//		float dx = curPoint.x - prevPoint.x;
		//		float dy = curPoint.y - prevPoint.y;
		//		PointF newPoint = PointF(x0 + t*dx, y0 + t*dy);
		//		sampledPolygon -> mPoints.push_back(newPoint);
		//		prevPoint = newPoint;
		//		curDist = 0;
		//	}
		//}
		//PointF firstPoint = sampledPolygon -> mPoints[0];
		//PointF lastPoint = sampledPolygon -> mPoints[sampledPolygon -> mPoints.size()];
		//float dx = lastPoint.x - firstPoint.x;
		//float dy = lastPoint.y - firstPoint.y;
		//float dist = sqrt(dx*dx + dy*dy);
		//if(dist < sampleDist*1.0)
		//{
		//	sampledPolygon -> mPoints.pop_back();
		//}
		PolygonF* sampledPolygon = normalPolygon(fgdPolygons[i], dist);
		result.push_back(sampledPolygon);
	}
	return result;
}
void GCApplication::dilateAndErode (Mat& con, int n)
{
	int niters = n;
	dilate(con, con, Mat(), Point(-1, -1), niters);
	erode(con, con, Mat(), Point(-1, -1), niters*2);
	dilate(con, con, Mat(), Point(-1, -1), niters);
}
Mat GCApplication::getKFrame(int k, VideoCapture& cap)
{
	Mat result;
	while(k--)
	{
		cap >> result;
	}
	return result;
}
GCApplication::GCApplication()
{
	mVideoSnapCut = new VideoSnapCut();
}


void GCApplication::showImageD()
{
	//Mat& tmp = resMask;
    if( rectDState == IN_PROCESS || rectDState == SET)
        rectangle( resMask, Point( rectD.x, rectD.y ), Point(rectD.x + rectD.width, rectD.y + rectD.height ), YELLOW, 1);
	
    imshow( "分割结果", resMask);
}

void GCApplication::showImageDwithRect()
{
	//Mat& tmp = resMask;
	Mat a;
	imageNext.copyTo(a);
	int n = rectDsPos.size();
	for(int i = 0; i < n; i++)
	{
       rectangle( a, Point( rectDsPos[i].x, rectDsPos[i].y ), Point(rectDsPos[i].x + rectDsPos[i].width, rectDsPos[i].y + rectDsPos[i].height ), RED, 1);

	}	
    imshow( "下一帧初始框", a);
}

void GCApplication::setRectsInMask()
{
	maskN.create( image->size(), CV_8UC1);
	mask.copyTo(maskN);
	for(int i = 0; i < rectDsPos.size(); i++)
	{
		rectDsPos[i].x = max(0, rectDsPos[i].x);
		rectDsPos[i].y = max(0, rectDsPos[i].y);
		rectDsPos[i].width = min(rectDsPos[i].width, image->cols-rectDsPos[i].x);
		rectDsPos[i].height = min(rectDsPos[i].height, image->rows-rectDsPos[i].y);
		(maskN(rectDsPos[i])).setTo( Scalar(GC_PR_FGD) );
	}

}


void GCApplication::setRect ( int event, int x, int y, int flags, void* )
{
    // TODO add bad args check
    switch( event )
    {
    case CV_EVENT_LBUTTONDOWN: // set rect or GC_BGD(GC_FGD) labels
        {            
                rectD = Rect( x, y, 1, 1 ); 
				rectDState = IN_PROCESS;
        }
        break;
    case CV_EVENT_LBUTTONUP:
		if( rectDState == IN_PROCESS )
		{
            rectD = Rect( Point(rectD.x, rectD.y), Point(x,y) );
			if(rectD.width >1 && rectD.height >1)
			{
				rectDs.push_back(rectD);
				rectDState = SET;
                showImageD();
			}
            
		}
	   break;
    case CV_EVENT_MOUSEMOVE:
        if( rectDState == IN_PROCESS )
        {
            rectD = Rect( Point(rectD.x, rectD.y), Point(x,y) );
            assert( bgdPxls.empty() && fgdPxls.empty() && prBgdPxls.empty() && prFgdPxls.empty() );
            //showImageD();
        }
        break;
    }
}

int GCApplication::nextFrameIter()
{
    grabCut( imageNext, maskN, rect, bgdModelN, fgdModelN, 1, GC_INIT_WITH_MASK );   
	Mat res;
    Mat binMask;
    getBinMask( maskN, binMask );
    imageNext.copyTo( res, binMask );
	res.copyTo(resMaskN);

    bgdPxlsN.clear(); fgdPxlsN.clear();
    prBgdPxlsN.clear(); prFgdPxlsN.clear();

    return iterCountN;
}

void GCApplication::snapcut()
{
	Mat con;
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
    resbinMask.copyTo(con);
	int niters = 0.1;
	dilate(con, con, Mat(), Point(-1, -1), niters);
	erode(con, con, Mat(), Point(-1, -1), niters*2);
	dilate(con, con, Mat(), Point(-1, -1), niters);
    findContours(con, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	vector<Point> poly; 
	vector<Rect> polyRect;
	int idx = 0;
	Mat result;
	image->copyTo(result);
	for(; idx >= 0; idx = hierarchy[idx][0])
	{
		approxPolyDP(Mat(contours[idx]), poly, 0.001, true);
	/*		vector<Point>::const_iterator itp = poly.begin(); 

	    while (itp != (poly.end() - 1)) 
		{ 
			line(result, *itp, *(itp + 1), Scalar(255), 1); 
			++itp; 
		} 
		line(result, *itp, *(poly.begin()), Scalar(255), 1);*/
		/*imshow("imageWithContour", result);
		waitKey(0);*/
	}
	/*imshow("imageWithContour", result);
	waitKey(0);*/
	//Mat a;
	//image.copyTo(a);
	//int dis = 10;
	////cout << poly.size() << endl;
	//for(int i = 0; i < poly.size(); i = i + 5)
	//{
	//	Point lu, rd;
	//	lu.x = max(poly[i].x - dis, 0);
	//	lu.y = max(poly[i].y - dis, 0);
	//	rd.x = min(poly[i].x + dis, result.cols - 1);
	//	rd.y = min(poly[i].y + dis, result.rows - 1);
	//	Rect r(lu, rd);
	//	polyRect.push_back(r);
	//	rectangle( result,lu, rd, YELLOW, 1);
	//	/*imshow( "上一帧初始轮廓框", result);
	//	waitKey(0);*/
	//}
	/*imshow( "上一帧初始轮廓框", result);
	waitKey(0);*/
	//waitKey(0);

	PolygonF* pyF = new PolygonF();
	for(int i = 0; i < poly.size(); i = i + 1)
	{

		PointF pt;
		pt.x = poly[i].x;
		pt.y = poly[i].y;
		pyF->mPoints.push_back(pt);
		//cout << "轮廓点的数目" << pyF.mPoints.size() << endl;
	}		
	ForegroundBorderPolygons.push_back(pyF);	//将边界多边形加入到容器中
	
	mVideoSnapCut->SetForegroundBorderPolygons(&ForegroundBorderPolygons);//从画笔中获得的轮廓多边形
	mVideoSnapCut->CreateMasks();
	mVideoSnapCut->BuildColorClassifiers();
	delete pyF;
	return ;
}

////预处理工具
void GCApplication::getBinMask( const Mat& comMask, Mat& binMask )
{
    if( comMask.empty() || comMask.type()!=CV_8UC1 )
        CV_Error( CV_StsBadArg, "comMask is empty or has incorrect type (not CV_8UC1)" );
    if( binMask.empty() || binMask.rows!=comMask.rows || binMask.cols!=comMask.cols )
        binMask.create( comMask.size(), CV_8UC1 );
    binMask = comMask & 1;
}

////光流显示函数////
void GCApplication::makecolorwheel(vector<Scalar> &colorwheel)  
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
  
void GCApplication::motionToColor(Mat flow, Mat &color)  
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