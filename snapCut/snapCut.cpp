#include <string>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "grabcutWithRect.h"
#include "VideoSnapCut.h"
#include "kcftracker.hpp"
#include <fstream>

using namespace std;
using namespace cv;

#define K 40                   ///从第几帧开始
//#define start 1

GCApplication gt;

static void help()
{
    cout << "\nThis program demonstrates GrabCut segmentation -- select an object in a region\n"
            "and then grabcut will attempt to segment it out.\n"
            "Call:\n"
            "./grabcut <image_name>\n"
			"\nSelect a rectangular area around the object you want to segment\n" <<
			"\nHot keys: \n"
			"\tESC - quit the program\n"
			"\tr - restore the original image\n"
			"\tn - next iteration\n"
			"\n"
			"\tleft mouse button - set rectangle\n"
			"\n"
			"\tCTRL+left mouse button - set GC_BGD pixels\n"
			"\tSHIFT+left mouse button - set CG_FGD pixels\n"
			"\n"
			"\tCTRL+right mouse button - set GC_PR_BGD pixels\n"
			"\tSHIFT+right mouse button - set CG_PR_FGD pixels\n" << endl;
}

void on_mouse( int event, int x, int y, int flags, void* param )
{
    gt.mouseClick( event, x, y, flags, param );
}
//void snapcut();
//PolygonF* pyF;
//

int main(void)
{
	/*参数设置模块*/
	//跟踪参数设置
	bool HOG = true;
	bool FIXEDWINDOW = true;
	bool MULTISCALE = false;
	bool SILENT = true;
	bool LAB = false;
	//输入模式参数设置：0表示从视频输入，1表示从帧序列输入
	bool inputFlag = true;
	
	Mat curFrame;
	if(inputFlag == false)
	{
		VideoCapture cap;
		cap.open("E:\\test\\03.avi");
		curFrame = gt.getKFrame(K, cap);  //输出第k帧后，cap保持从第k帧继续输出
	}
	
	
	//关键帧标志
	bool flag = false;       
	bool fromKeyFlag = true;

	int start = 1;
	int count  = start;      //帧编号
    char* name = new char[100];
	char* path = "E:\\snapcut\\input\\images\\%04d.jpg";//"E:\\videoCut\\SegTrackv2\\SegTrackv2\\JPEGImages\\worm\\%010d.png";   // "E:\\videoCut\\SegTrackv2\\SegTrackv2\\JPEGImages\\bird_of_paradise\\bird_of_paradise_%05d.png";//"E:\\videoCut\\SegTrackv2\\SegTrackv2\\JPEGImages\\parachute\\parachute_%05d.png";//"E:\\videoCut\\SegTrackv2\\SegTrackv2\\JPEGImages\\monkey\\%010d.png"; // "E:\\test\\images\\%04d.jpg";   //"E:\\videoCut\\SegTrackv2\\SegTrackv2\\JPEGImages\\monkey\\%010d.png";	
	
	const string keyFrameWin = "第一帧或关键帧交互窗口";
	
	while(1)  //视频分割循环
	{

		if(count == start || flag == true) //第一帧或关键帧时进行的处理
		{
			gt.mVideoSnapCut->setFrameNum(count);
			if(inputFlag)
			{
				std::sprintf(name, path, count);  
				curFrame = imread(name);
			}

			gt.mVideoSnapCut -> SetCurFrameMat(curFrame);
			gt.mVideoSnapCut -> SetCurFrame(count);

			cv::namedWindow(keyFrameWin, WINDOW_AUTOSIZE);    //关键帧交互
			gt.setImageAndWinName(curFrame, keyFrameWin);
			setMouseCallback(keyFrameWin, on_mouse, 0);
			gt.showImage();

			for(;;)       //对关键帧的多次交互分割
			{
				int c = waitKey(0);
				if(c == 'r')
				{
					cout <<"标记错误，请重置："<< endl;
					gt.reset();           //前景和背景标记像素容器清零，三类标记清零，迭代次数清零
					//gt.showJiaoHuImage();       //展示图像是为了在图像上标记种子点
					gt.showImage();
				}
				else if(c == 'n')
				{
					int iterCount = gt.getIterCount();
					cout << "<关键帧第" << iterCount << "次分割... ";
					int newIterCount = gt.nextIter(); //关键帧分割完成，返回分割次数
					if( newIterCount > iterCount )
					{
						//gt.saveCutResult();
						gt.showImage();
						Mat tmp = gt.resMask;         //得到彩色分割结果
						Mat tmp1 = gt.resbinMask ;    //得到0、1二值分割结果
						imshow("分割结果", tmp);					
						cout << iterCount << "完成>" << endl;
					}
					else
						cout << "rect must be determined>" << endl;
				}
				else if(c == 's')  //进入下一帧的snapcut分割
				{
					count++;        //当前帧分割好了进入后续帧
					flag = false;   //snapcut分割的帧不是关键帧
					fromKeyFlag = true;    //上一帧是关键帧标志位
					break;					
				}			
			}
		}
		else //非关键帧的处理
		{
			cout << "第" << count << "帧snapcut分割开始" << endl;
			gt.mVideoSnapCut->setFrameNum(count);
			Mat con , combine;
			//获得轮廓的值矩阵
			if(fromKeyFlag)					//上一帧是关键帧
			{
				gt.resbinMask.copyTo(con);  //0或1的二值矩阵
				fromKeyFlag = false;
			}				
			else                            //上一帧不是关键帧
				gt.mVideoSnapCut->nextContourMat.copyTo(con);
			con.copyTo(combine);  		
			con.copyTo(gt.mVideoSnapCut->binMask1);//将分割二值结果传递到snapcut中
			gt.mVideoSnapCut -> generateContourWithRect(con);

			vector<Rect>polyRect;			
			Mat nextFrame, updateTrack;
			curFrame.release();

			sprintf(name, path, count);//读入新的帧
		    curFrame = imread(name);
			curFrame.copyTo(nextFrame);
			curFrame.copyTo(updateTrack);                     //当前帧作为更新局部跟踪框的帧

			//vector<Rect> updateRects;
			//for(int i = 0; i < num - 1; i++)
			//{
			//	Rect updateRect = trackers[i].update(curFrame);
			//	updateRects.push_back(updateRect);
			//	//rectangle( updateTrack, Point( updateRect.x, updateRect.y ), Point( updateRect.x+updateRect.width, updateRect.y+updateRect.height), Scalar( 0, 0, 255 ), 1, 8 );
			//}
			//vector<Rect> updateRectsNew;
			//gt.updateRects(updateRects, updateRectsNew);
			//Mat updateTrackNew;
			//updateTrack.copyTo(updateTrackNew);
			//
			//gt.showRects(updateTrack, updateRects);
			//imshow("局部跟踪结果",updateTrack);
			//gt.showRects(updateTrackNew, updateRectsNew);
			//imshow("局部跟踪结果均匀化",updateTrackNew);
			//waitKey();
			//第一阶段调整，局部跟踪框相对于全局跟踪框的位置变化不应该太大
			//第二阶段跟踪框的调整采用canny算子
			//Mat canny = gt.getCanny(curFrame);
			//waitKey();
			//调整跟踪框的位置
			/*for(int i = 0; i < num - 1; i++)
			{

			}*/

			//对整体跟踪框内的图像做边缘检测

					
			/*gt.mVideoSnapCut -> SetCurFrameMat(tmp);
			gt.mVideoSnapCut -> SetCurFrame(count);*/
			gt.mVideoSnapCut -> CreateMasks();
			gt.mVideoSnapCut -> mCombinedMask = combine;
			gt.mVideoSnapCut -> RegisterCurFrameWithFrame(nextFrame);
			vector<Mat> vM = gt.mVideoSnapCut -> mAfterTransContours;      //register之后
			vector<vector<Point>> vvP;
			Mat tmpNextFrame;
			nextFrame.copyTo(tmpNextFrame);

			vector<PolygonF*> ForegroundBorderPolygons;
			for(int i = 0; i < vM.size(); i++)
			{
				gt.transMatTocontours(vM[i], vvP);
				for(int j = 0; j < vvP.size(); j++)
				{
					PolygonF* pyF = gt.showContour(vvP[j], tmpNextFrame, VideoSnapCut::mMaskSize*2.0f/3, polyRect, 1);					
					ForegroundBorderPolygons.push_back(pyF);	//将边界返回的所有轮廓上的点加入到容器中
					imshow("仿射变换后的均匀采样框", tmpNextFrame);

					char* path1 = "E:\\shiyanResult\\第%d帧仿射变换后的均匀采样框.jpg";//存储仿射变换后的均匀采样框
					char* name1 = new char[100];
					sprintf(name1, path1, count);
					imwrite(name1, tmpNextFrame);
					delete[] name1;
				}
			}
			gt.mVideoSnapCut -> SetForegroundBorderPolygons(&ForegroundBorderPolygons);		
			gt.mVideoSnapCut -> BuildColorClassifiers(); 
			gt.mVideoSnapCut -> ClassifyPixels();
			gt.mVideoSnapCut -> SetCurFrameMat(nextFrame);
			fromKeyFlag = false;
			count++;
			waitKey();
		}		
	}

	return 0;
}