#include "stdafx.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "kcftracker.hpp"

//#include <dirent.h>

using namespace std;
using namespace cv;

bool HOG = true;
bool FIXEDWINDOW = true;
bool MULTISCALE = false;
bool SILENT = true;
bool LAB = false;

KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);
static void on_mouse( int event, int x, int y, int flags, void* param )
{
    tracker.mouseClick( event, x, y, flags, param );
}

int main1(void){

	VideoCapture cap;
	cap.open("E:\\testc\\media\\media22.mov");

	// Frame readed
	Mat frame;

	// Tracker results
	Rect result;
	// Frame counter
	int nFrames = 0;
	 
	Mat startF;
	cap >> startF;
	/*int k = 5;
	
	while(1)
	{
		vector<Mat> vm;
		for(int i = 0; i < 5; i++)
		{
			cap >> startF;
			vm.push_back(startF);
		}	
		break;
	}
	cout << "程序终止" << endl;
	return 0;*/
	startF.copyTo(tracker.cur);
    const string winName = "tracker";
    namedWindow( winName);
	imshow(winName, startF);
    setMouseCallback( winName, on_mouse, 0 );
	
	waitKey(0);
	cout << "总共有：" << tracker.rects.size() << "个跟踪框" << endl;
	int n = tracker.rects.size();
	/*vector<KCFTracker> trackers;
	for(int i = 0; i < n; i++)
	{
		KCFTracker track(HOG, FIXEDWINDOW, MULTISCALE, LAB);
		trackers.push_back(track);
	}*/
	vector<KCFTracker> trackers(n, KCFTracker(HOG, FIXEDWINDOW, MULTISCALE, LAB));

	while(1)
	{
		
		if(nFrames == 0)
		{
			startF.copyTo(frame);

			for(int i = 0; i < n; i++)
			{
				Rect r = tracker.rects[i];
				trackers[i].init( r, frame );
				rectangle( frame, r, Scalar( 0, 255, 255 ), 1, 8 );
			}			
			imshow("tracker", frame);
			waitKey(30);
		}
		else   // 更新
		{
			for(int i = 0; i < n; i++)
			{
				result = trackers[i].update(frame);
			    rectangle( frame, Point( result.x, result.y ), Point( result.x+result.width, result.y+result.height), Scalar( 0, 0, 255 ), 1, 8 );
			}
		    imshow("tracker", frame);
			waitKey(30);
		}
		cap >> frame;
		if(frame.empty())
			break;
		nFrames++;
		//if(waitKey(30) >= 0) break;
	}
	return 0;
}