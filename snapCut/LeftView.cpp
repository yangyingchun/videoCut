// InterleavedVideoCutOutView.cpp : implementation of the CLeftView class
//
#include "stdafx.h"
#include "assert.h"
#include "time.h"
#include "stdlib.h"
#include "App.h"

#include "Doc.h"
#include "LeftView.h"
#include "RightView.h"
#include "BMPImage.h"

#include "FolderDlg.h"
#include "grabcutWithRect.h"

#ifdef _DEBUG
#define new DEBUG_NEW 
#endif


// CLeftView

IMPLEMENT_DYNCREATE(CLeftView, CFormView)

BEGIN_MESSAGE_MAP(CLeftView, CFormView)
	ON_BN_CLICKED(IDC_BUTTONOPEN, &CLeftView::OnBnClickedButtonOpen)
	ON_BN_CLICKED(IDC_BUTTONSELNONE, &CLeftView::OnBnClickedButtonselnone)
	ON_BN_CLICKED(IDC_BUTTONCLEARFRAMES, &CLeftView::OnBnClickedButtonclearframes)
	ON_BN_CLICKED(IDC_BUTTONSELFGMASK, &CLeftView::OnBnClickedButtonselfgmask)
	ON_BN_CLICKED(IDC_BUTTONSAMPLECONTOUR, &CLeftView::OnBnClickedButtonSampleContour)
	ON_BN_CLICKED(IDC_BUTTONMASK, &CLeftView::OnBnClickedButtonCreateMask)
	ON_BN_CLICKED(IDC_BUTTONDISTANCETRANSFORM, &CLeftView::OnBnClickedButtonDistanceTransform)
	ON_BN_CLICKED(IDC_BUTTONPROBABILITY, &CLeftView::OnBnClickedButtonProbability)
	ON_BN_CLICKED(IDC_BUTTONEXPLORE, &CLeftView::OnBnClickedButtonexplore)
	ON_BN_CLICKED(IDC_BUTTONFIRST, &CLeftView::OnBnClickedButtonfirst)
	ON_BN_CLICKED(IDC_BUTTONPREV, &CLeftView::OnBnClickedButtonprev)
	ON_BN_CLICKED(IDC_BUTTONNEXT, &CLeftView::OnBnClickedButtonnext)
	ON_BN_CLICKED(IDC_BUTTONLAST, &CLeftView::OnBnClickedButtonlast)
	ON_BN_CLICKED(IDC_BUTTONGO, &CLeftView::OnBnClickedButtongo)
	ON_BN_CLICKED(IDC_BUTTONREGISTER, &CLeftView::OnBnClickedButtonregister)
	ON_BN_CLICKED(IDC_BUTTONLOAD, &CLeftView::OnBnClickedButtonload)

	ON_BN_CLICKED(IDC_BUTTONKEYFRAMECUT, &CLeftView::OnBnClickedButtonKeyframeCut)

END_MESSAGE_MAP()

// CLeftView construction/destruction
//LSApplication mLazySnapping;
//
//static void on_mouse(int event, int x, int y, int flags, void* param) {
//    mLazySnapping.mouseClick(event, x, y, flags, param);
//}
GCApplication gcapp;

static void on_mouse( int event, int x, int y, int flags, void* param )
{
    gcapp.mouseClick( event, x, y, flags, param );
}

CLeftView::CLeftView()
	: CFormView(CLeftView::IDD)
	, mFilePath(_T(""))
	, mEdtNumberFrames(0)
	, mEdtCurrentFrame(0)
{
	// TODO: add construction code here
	((CApp*)AfxGetApp())->mLeftView = this;
	mSelectionMode = 0;
	srand((uint)time(0));
	
	mImageSegmented = 0;


	mVideoSnapCut = 0;
	mVideoSnapCut = new VideoSnapCut();

}

CLeftView::~CLeftView()
{	
	ClearForegroundBorderPolygons();

	
}

void CLeftView::ClearForegroundBorderPolygons()
{
	for(uint i=0;i<mForegroundBorderPolygons.size();i++)
	{
		delete mForegroundBorderPolygons[i];
	}
	mForegroundBorderPolygons.clear();
}

void CLeftView::DoDataExchange(CDataExchange* pDX)
{
	CFormView::DoDataExchange(pDX);
	DDX_Text(pDX, IDC_EDITPATH, mFilePath);
	DDX_Text(pDX, IDC_EDITNUMBERFRAMES, mEdtNumberFrames);
	DDX_Text(pDX, IDC_EDITCURRENTRAME, mEdtCurrentFrame);
}

BOOL CLeftView::PreCreateWindow(CREATESTRUCT& cs)
{
	// TODO: Modify the Window class or styles here by modifying
	//  the CREATESTRUCT cs

	return CFormView::PreCreateWindow(cs);
}

void CLeftView::OnInitialUpdate()
{
	CFormView::OnInitialUpdate();
	GetParentFrame()->RecalcLayout();
	ResizeParentToFit();
	mFilePath = "E:\\Dev\\InterleavedVideoCutOut\\images\\snapcut2.bmp";
	//mFilePath = "C:\\Papers\\SiggraphAsis12\\Zhong12-SIGA-dataset\\HUMAN\\fq\\000.jpg";
	mFilePath = "E:\\Dev\\InterleavedVideoCutOut\\images\\videosnapcut.mp4";
	mMediaMode = 2;
	if(1)
	{
		mFilePath = "E:\\Projects\\images\\ANIMAL\\bear";
		mMediaMode = 3;
		mFilesPath.clear();
		ListFiles(mFilesPath);
	}
	UpdateData(0);

	
	LoadImageFile();

}

// CLeftView diagnostics

#ifdef _DEBUG
void CLeftView::AssertValid() const
{
	CFormView::AssertValid();
}

void CLeftView::Dump(CDumpContext& dc) const
{
	CFormView::Dump(dc);
}

CDoc* CLeftView::GetDocument() const // non-debug version is inline
{
	ASSERT(m_pDocument->IsKindOf(RUNTIME_CLASS(CDoc)));
	return (CDoc*)m_pDocument;
}
#endif //_DEBUG

// CLeftView message handlers

void CLeftView::OnBnClickedButtonOpen()
{



	 CFileDialog dlg(TRUE,
		 NULL,NULL,OFN_OVERWRITEPROMPT,
		 //"Bitmap Files (*.bmp)|*.bmp|"
		 //"Avi Files (*.avi)|*.avi|"
		 //"MP4 Files (*.mp4)|*.mp4|"
		 //"All Files(*.*)|*.*||",
		 "Multimedia Files (*.bmp; *.jpg; *.png; *.avi; *.mp4)|*.bmp; *.jpg; *.png; *.avi; *.mp4|"
		 "Image Files (*.bmp; *.jpg; *.png)|*.bmp; *.jpg; *.png|"
		 "Video Files (*.avi; *.mp4)|*.avi; *.mp4|"
		 "All Files(*.*)|*.*||",		 
		 this);

	 
	 
	if (dlg.DoModal() == IDOK)
	{
		CString fileName = dlg.GetFileName();
		//CString folderName = dlg.GetFolderPath();
		mFilePath = dlg.GetPathName();
		
		int extIndex = mFilePath.ReverseFind('.');

		if(extIndex != -1) // file
		{
			int len = mFilePath.GetLength();	
			CString extension = mFilePath.Mid(extIndex+1,len-(extIndex+1));

			if( extension.CompareNoCase("bmp") == 0 || 
				extension.CompareNoCase("jpg") == 0 || 
				extension.CompareNoCase("png") == 0 || 
				extension.CompareNoCase("ppm") == 0 )
			{
				mMediaMode = 1;//Ϊ1��ʱ����ͼƬ
			}
			else if(extension.CompareNoCase("avi") == 0 || 
			extension.CompareNoCase("mp4") == 0)
			{
				mMediaMode = 2;//Ϊ0��ʱ������Ƶ
			}
		}
		

		UpdateData(0);

		//LoadImageFile();

		((CApp*)AfxGetApp())->mRightView->Invalidate();
	}
}

int CLeftView::ListFiles(vector<string> &files)
{
	WIN32_FIND_DATA FindFileData;
	HANDLE hFind = INVALID_HANDLE_VALUE;
	int Error;

	// Find the first file in the directory.
	CString allFilesPath = mFilePath + "\\*";
	hFind = FindFirstFile(allFilesPath, &FindFileData);

	if (hFind == INVALID_HANDLE_VALUE)
	{
		//printf ("Invalid file handle. Error is %u.\n", GetLastError());
		return 0;
	}
	else
	{
		// List all the other files in the directory.
		do 
		{
			if (FindFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
			{
				//If it's a directory nothing should happen. Just continue with the next file.
			}
			else
			{
				CString fileName(FindFileData.cFileName);
				if(fileName.Find(".bmp") != -1 || fileName.Find(".png") != -1 || fileName.Find(".jpg") != -1)
				{
					CString filePath = mFilePath + "\\" + fileName;
					//AfxMessageBox(filePath);
					files.push_back((LPCTSTR)filePath);
				}
				
			}

		}while (FindNextFile(hFind, &FindFileData) != 0);

		Error = GetLastError();
		FindClose(hFind);
		if (Error != ERROR_NO_MORE_FILES)
		{
			//printf ("FindNextFile error. Error is %u.\n", dwError);
			return 0;
		}
	}

	return 1;
}

void CLeftView::OnBnClickedButtonexplore()
{
	CFolderDialog dlg(  _T( "Select Images Folder" ), _T( "E:\\Dev\\InterleavedVideoCutOut\\images" ), this );
    if( dlg.DoModal() == IDOK  )
    {    
		mFilePath  = dlg.GetFolderPath();
		UpdateData(0);
		mFilesPath.clear();
		ListFiles(mFilesPath);
		mMediaMode = 3;
    }
}

void CLeftView::OnBnClickedButtonLoad()
{
	LoadImageFile();
}

void CLeftView::SetImageMat(const Mat& mFrameMat)//��mat�����ΪIPLImage
{
	ClearForegroundBorderPolygons();

	int width = mFrameMat.cols;
	int height = mFrameMat.rows;

	mVideoSnapCut->SetCurFrameMat(mFrameMat);

	uchar *tempData =  new uchar[width*height*3];

	for (int y = 0; y < height; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
			Vec3b intensity = mFrameMat.at<Vec3b>(y, x);
			uchar blue = intensity.val[0];
			uchar green = intensity.val[1];
			uchar red = intensity.val[2];
			int index1 = (y*width+x)*3;
			tempData[index1+0] = blue;
			tempData[index1+1] = green;
			tempData[index1+2] = red;
		}
	}


	((CApp*)AfxGetApp())->mRightView->SetImageData(width, height, tempData, (mImageSegmented != 1));	
	//��IPLImage��������ͼ
	delete []tempData;
	

	UpdateData(0);
}

void CLeftView::LoadImageFromFile(string fileName)
{
	
	Mat mFrameMat = imread(fileName.data());
	
	if( !mFrameMat.empty() )
	{
		SetImageMat(mFrameMat);
		mVideoSnapCut->SetCurFrameMat(mFrameMat);	
		/*mLazySnapping.images = mFrameMat.clone();
		mLazySnapping.image = &mFrameMat;*/
	}

}

void CLeftView::LoadVideoFromFile(string fileName)
{	
	mVideoReader.Close();

	mVideoReader.Open(fileName);
		
	Mat mFrameMat;
	if(mVideoReader.GoToFirstFrame(mFrameMat))
	{
		if( !mFrameMat.empty() )
		{
			SetImageMat(mFrameMat);
			mVideoSnapCut->SetCurFrameMat(mFrameMat);
			/*mLazySnapping.images = mFrameMat.clone();
		    mLazySnapping.image = &mFrameMat;*/
		}
	}
}

void CLeftView::LoadImageFile()//����ִ��
{
	UpdateData();	
	
	int extIndex = mFilePath.ReverseFind('.');
	
	
	if(extIndex != -1) // file
	{
		int len = mFilePath.GetLength();	
		CString extension = mFilePath.Mid(extIndex+1,len-(extIndex+1));

		if( extension.CompareNoCase("bmp") == 0 || 
			extension.CompareNoCase("jpg") == 0 || 
			extension.CompareNoCase("png") == 0 || 
			extension.CompareNoCase("ppm") == 0 )
		{
			LoadImageFromFile((LPCSTR)mFilePath);

			mEdtNumberFrames = 1;
			mEdtCurrentFrame = 1;
			UpdateData(0);
		}
		else if(extension.CompareNoCase("avi") == 0 || 
			extension.CompareNoCase("mp4") == 0)
		{
			LoadVideoFromFile((LPCSTR)mFilePath);
			mEdtNumberFrames = mVideoReader.mNumFrames;
			mCurFrame = mVideoReader.mCurFrame-1;
			mEdtCurrentFrame = mCurFrame+1;
			UpdateData(0);			
			mVideoSnapCut->SetVideoReader(&mVideoReader);
			mVideoSnapCut->SetCurFrame(mCurFrame);
		}
	}
	else // folder
	{
		if(mFilesPath.size()>0)
		{
			mCurFrame = 0;
			mEdtNumberFrames = mFilesPath.size();			
			mEdtCurrentFrame = mCurFrame + 1;
			UpdateData(0);

			LoadImageFromFile(mFilesPath[mCurFrame]);
			
			mVideoSnapCut->SetFilesPath(&mFilesPath);//��·��

			mVideoSnapCut->SetCurFrame(mCurFrame);//����ǰ֡���
		}//if(mFilesPath.size()>0)

	}//else // folder
}

void CLeftView::AddForegroundBorder(PolygonF *borderPolygon)
{
	if(!borderPolygon)
	{
		return;
	}
	mForegroundBorderPolygons.push_back(borderPolygon);	//���߽����μ��뵽������
}

void CLeftView::OnBnClickedButtonselnone()
{
	mSelectionMode = 0;
}

void CLeftView::OnBnClickedButtonclearframes()
{
	if(mVideoSnapCut)
	{
		delete mVideoSnapCut;
		mVideoSnapCut = new VideoSnapCut();
	}

	
	((CApp*)AfxGetApp())->mRightView->ClearScreen();
}

void CLeftView::OnBnClickedButtonselfgmask()
{
	mSelectionMode = 3;
}

void CLeftView::OnBnClickedButtonSampleContour()
{

	mVideoSnapCut->SetForegroundBorderPolygons(&mForegroundBorderPolygons);
	mVideoSnapCut->CreateMasks();//֮ǰȥ����
	mVideoSnapCut->ResampleContour();
	
	((CApp*)AfxGetApp())->mRightView->mDrawBorder = true;
	((CApp*)AfxGetApp())->mRightView->Invalidate();
	
}

void CLeftView::OnBnClickedButtonCreateMask()
{
	mVideoSnapCut->SetForegroundBorderPolygons(&mForegroundBorderPolygons);//��leftview�е������ߴ���snapcut��
	mVideoSnapCut->CreateMasks();

	Mat mFrameMat = mVideoSnapCut->mImage;
	uchar *tempData =  new uchar[mFrameMat.cols*mFrameMat.rows*3];

	for(int r=0;r<mFrameMat.rows;r++)
	{
		for(int c = 0;c<mFrameMat.cols;c++)
		{
			int index1 = (r*mFrameMat.cols+c)*3;			
			Vec3b intensity = mFrameMat.at<Vec3b>(r, c);
			Color pix(intensity.val[2],intensity.val[1],intensity.val[0]);

			//pix.r *= 255; pix.r = pix.r < 0 ? 0 : pix.r > 255 ? 255 : pix.r;
			//pix.g *= 255; pix.g = pix.g < 0 ? 0 : pix.g > 255 ? 255 : pix.g;
			//pix.b *= 255; pix.b = pix.b < 0 ? 0 : pix.b > 255 ? 255 : pix.b;
			if(mVideoSnapCut->mCombinedMask.at<uchar>(r,c) == 1)
			{
				float t1 = 0.5, t2 = 1-t1;
				tempData[index1+0] = (uchar)(t1*255 + t2*pix.b);
				tempData[index1+1] = (uchar)(t1*255 + t2*pix.g);
				tempData[index1+2] = (uchar)(t1*255 + t2*pix.r);
				
			}
			else 
			{
				float t1 = 0.5, t2 = 1-t1;
				tempData[index1+0] = (uchar)(t1*0 + t2*pix.b);
				tempData[index1+1] = (uchar)(t1*0 + t2*pix.g);
				tempData[index1+2] = (uchar)(t1*0 + t2*pix.r);
			}
			
		}
	}
	((CApp*)AfxGetApp())->mRightView->SetImageData(mFrameMat.cols, mFrameMat.rows, tempData, false);		
	delete []tempData;

	imwrite("e:\\CombinedMask.jpg", mVideoSnapCut->mCombinedMask);//֮ǰȥ����

}

bool CLeftView::pointInPolygon(float x, float y) 
{
	bool oddNodes=false;
	for(uint ii=0;ii<mForegroundBorderPolygons.size();ii++)
	{
		int polygonSize = mForegroundBorderPolygons[ii]->mPoints.size();
		if(polygonSize<2)
		{
			continue;
		}
		PolygonF* polygon = mForegroundBorderPolygons[ii];
		
		
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
	}
	return oddNodes;
}

void CLeftView::OnBnClickedButtonDistanceTransform()
{
	#define INF 1000000000
	mVideoSnapCut->SetForegroundBorderPolygons(&mForegroundBorderPolygons);
	mVideoSnapCut->CreateMasks();
	
	int randImageIndex =  rand()%mVideoSnapCut->mDistanceTransforms.size();
	Mat distTransform = mVideoSnapCut->mDistanceTransforms[randImageIndex];
	float maxDist = 0;
	 
	Mat mFrameMat = mVideoSnapCut->mImage;

	for(int r=0;r<mFrameMat.rows;r++)
	{
		for(int c = 0;c<mFrameMat.cols;c++)
		{
			if(distTransform.at<float>(r,c)>maxDist)
			{
				maxDist = distTransform.at<float>(r,c);
				if(maxDist == INF)
				{
					maxDist = maxDist;
				}
			}
		}
	}

	assert(maxDist != INF && maxDist != 0);
	
	uchar *tempData =  new uchar[mFrameMat.cols*mFrameMat.rows*3];

	for(int r=0;r<mFrameMat.rows;r++)
	{
		for(int c = 0;c<mFrameMat.cols;c++)
		{
			int index1 = (r*mFrameMat.cols+c)*3;			
			Vec3b intensity = mFrameMat.at<Vec3b>(r, c);
			Color pix(intensity.val[2],intensity.val[1],intensity.val[0]);
			//pix.r *= 255; pix.r = pix.r < 0 ? 0 : pix.r > 255 ? 255 : pix.r;
			//pix.g *= 255; pix.g = pix.g < 0 ? 0 : pix.g > 255 ? 255 : pix.g;
			//pix.b *= 255; pix.b = pix.b < 0 ? 0 : pix.b > 255 ? 255 : pix.b;
			float distNorm = distTransform.at<float>(r,c)/maxDist;
			int colorDistNorm = (int)(distNorm * 255);
			Color distPix((float)colorDistNorm, (float)colorDistNorm, (float)colorDistNorm);
				
		
			float t1 = 0.9f, t2 = 1-t1;
			tempData[index1+0] = (uchar)(t1*distPix.b + t2*pix.b);
			tempData[index1+1] = (uchar)(t1*distPix.g + t2*pix.g);
			tempData[index1+2] = (uchar)(t1*distPix.r + t2*pix.r);
			
			
		}
	}
	((CApp*)AfxGetApp())->mRightView->SetImageData(mFrameMat.cols, mFrameMat.rows, tempData, false);		
	delete []tempData;
}



void CLeftView::OnBnClickedButtonfirst()
{	
	if(mMediaMode == 1)
	{
		return;
	}
	else if(mMediaMode == 2)
	{
		Mat mFrameMat;

		mVideoReader.GoToFirstFrame(mFrameMat);
		mCurFrame = mVideoReader.mCurFrame-1;
		mEdtCurrentFrame = mCurFrame+1;
		UpdateData(0);	
		if( !mFrameMat.empty() )
		{
			SetImageMat(mFrameMat);
			mVideoSnapCut->SetCurFrameMat(mFrameMat);
			/*mLazySnapping.images = mFrameMat.clone();
		    mLazySnapping.image = &mFrameMat;*/
		}
		mVideoSnapCut->SetCurFrame(mCurFrame);
	}
	else if(mMediaMode == 3)
	{
		mCurFrame = 0;
		mEdtCurrentFrame = mCurFrame + 1;
		UpdateData(0);
		LoadImageFromFile(mFilesPath[mCurFrame]);	
		mVideoSnapCut->SetCurFrame(mCurFrame);
	}
}

void CLeftView::OnBnClickedButtonprev()
{
	if(mMediaMode == 1)
	{
		return;
	}
	else if(mMediaMode == 2)
	{
		Mat mFrameMat;

		mVideoReader.GetPrevFrame(mFrameMat);
		mCurFrame = mVideoReader.mCurFrame-1;
		mEdtCurrentFrame = mCurFrame+1;
		UpdateData(0);	
		if( !mFrameMat.empty() )
		{
			SetImageMat(mFrameMat);
			mVideoSnapCut->SetCurFrameMat(mFrameMat);
			/*mLazySnapping.images = mFrameMat.clone();
	    	mLazySnapping.image = &mFrameMat;*/
		}
		mVideoSnapCut->SetCurFrame(mCurFrame);
	
	}
	else if(mMediaMode == 3)
	{
		if(mCurFrame > 0)
		{
			mCurFrame--;
			mEdtCurrentFrame = mCurFrame + 1;
			UpdateData(0);
			LoadImageFromFile(mFilesPath[mCurFrame]);
			mVideoSnapCut->SetCurFrame(mCurFrame);
		}
	}
}

void CLeftView::OnBnClickedButtonnext()
{
	if(mMediaMode == 1)
	{
		return;
	}
	else if(mMediaMode == 2)
	{
		Mat mFrameMat;
		mVideoReader.GetNextFrame(mFrameMat);
		mCurFrame = mVideoReader.mCurFrame-1;
		mEdtCurrentFrame = mCurFrame+1;
		UpdateData(0);	
		if( !mFrameMat.empty() )
		{
			SetImageMat(mFrameMat);
			mVideoSnapCut->SetCurFrameMat(mFrameMat);
			/*mLazySnapping.images = mFrameMat.clone();
	     	mLazySnapping.image = &mFrameMat;*/
		}
		mVideoSnapCut->SetCurFrame(mCurFrame);
	
	}
	else if(mMediaMode == 3)
	{
		if(mFilesPath.size()>0)
		{
			if((uint)mCurFrame < mFilesPath.size()-1)
			{
				mCurFrame++;
				mEdtCurrentFrame = mCurFrame + 1;
				UpdateData(0);
				LoadImageFromFile(mFilesPath[mCurFrame]);
				mVideoSnapCut->SetCurFrame(mCurFrame);
			}
		}
	}
}

void CLeftView::OnBnClickedButtonlast()
{
	if(mMediaMode == 1)
	{
		return;
	}
	else if(mMediaMode == 2)
	{
		Mat mFrameMat;
		mVideoReader.GoToLastFrame(mFrameMat);
		mCurFrame = mVideoReader.mCurFrame-1;
		mEdtCurrentFrame = mCurFrame+1;
		UpdateData(0);	
		if( !mFrameMat.empty() )
		{
			SetImageMat(mFrameMat);
			mVideoSnapCut->SetCurFrameMat(mFrameMat);
			/*mLazySnapping.images = mFrameMat.clone();
		    mLazySnapping.image = &mFrameMat;*/
		}
		mVideoSnapCut->SetCurFrame(mCurFrame);
	
	}
	else if(mMediaMode == 3)
	{
		if(mFilesPath.size()>0)
		{
			mCurFrame = mFilesPath.size()-1;
			mEdtCurrentFrame = mCurFrame + 1;
			UpdateData(0);
			LoadImageFromFile(mFilesPath[mCurFrame]);
			mVideoSnapCut->SetCurFrame(mCurFrame);
		}		
	}
}

void CLeftView::OnBnClickedButtongo()
{
	UpdateData();
	mCurFrame = mEdtCurrentFrame-1;
	mCurFrame = mCurFrame<0?0:mCurFrame;
	if(mMediaMode == 1)
	{
		return;
	}
	else if(mMediaMode == 2)
	{
		Mat mFrameMat;
		mVideoReader.GoToFrame(mCurFrame,mFrameMat);
		mCurFrame = mVideoReader.mCurFrame-1;
		mEdtCurrentFrame = mCurFrame+1;
		UpdateData(0);	
		if( !mFrameMat.empty() )
		{
			SetImageMat(mFrameMat);
			mVideoSnapCut->SetCurFrameMat(mFrameMat);
		}
		mVideoSnapCut->SetCurFrame(mCurFrame);

	
	}
	else if(mMediaMode == 3)
	{
		if(mFilesPath.size()>0)
		{
			mCurFrame = (uint)mCurFrame>mFilesPath.size()-1?mFilesPath.size()-1:mCurFrame;
			LoadImageFromFile(mFilesPath[mCurFrame]);
			mVideoSnapCut->SetCurFrame(mCurFrame);
		}
		
	}
	mEdtCurrentFrame = mCurFrame + 1;
	UpdateData(0);
}

void CLeftView::OnBnClickedButtonProbability()
{
	mVideoSnapCut->SetForegroundBorderPolygons(&mForegroundBorderPolygons);//�ӻ����л�õ����������
	mVideoSnapCut->CreateMasks();
	mVideoSnapCut->BuildColorClassifiers();
	mVideoSnapCut->ClassifyPixels();
	mVideoSnapCut->ResampleContour();
	((CApp*)AfxGetApp())->mRightView->Invalidate();
}

void CLeftView::OnBnClickedButtonregister()
{
	mVideoSnapCut->RegisterCurFrameWithFrame(mCurFrame+1);
	mVideoSnapCut->BuildColorClassifiers();
	mVideoSnapCut->ClassifyPixels();
	mVideoSnapCut->generateContourWithRect(mVideoSnapCut->binMask);
	mVideoSnapCut->ResampleContour();
	((CApp*)AfxGetApp())->mRightView->Invalidate();

	mCurFrame = mVideoSnapCut->mCurFrame;
	mEdtCurrentFrame = mCurFrame+1;
	UpdateData(0);	//��������ֵд��ռ���
}

void CLeftView::OnBnClickedButtonload()
{
	LoadImageFile();
}

void CLeftView::OnBnClickedButtonKeyframeCut()
{
    const string winName = "grabcutWithRect";
	setMouseCallback( winName, on_mouse, 0 );
	Mat tmp;
	mVideoSnapCut->mImage.copyTo(tmp);
    gcapp.setImageAndWinName( tmp, winName );
    gcapp.showImage();

    for(;;)
    {
        int c = waitKey(0);
        switch( (char) c )
        {
        case '\x1b':
            cout << "Exiting ..." << endl;
            goto exit_main;
		case 'd':
			goto exit_cut;
        case 'r':
            cout << endl;
            gcapp.reset();
            gcapp.showImage();
            break;
        case 'n':
            int iterCount = gcapp.getIterCount();
            cout << "<" << iterCount << "... ";
            int newIterCount = gcapp.nextIter();
            if( newIterCount > iterCount )
            {
                gcapp.showImage();
				Mat tmp = gcapp.resMask;
				Mat tmp1 = gcapp.resbinMask ;
				imshow("�ָ���", tmp);
				imwrite("I:\\test\\resbinMask.jpg",tmp1);
				imwrite("I:\\test\\resMask.jpg",tmp);
                cout << iterCount << ">" << endl;
            }
            else
                cout << "rect must be determined>" << endl;
            break;
		
        }

    }

exit_main:
    destroyWindow( winName );

	////��������////
exit_cut:
	Mat con;
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
    gcapp.resbinMask.copyTo(con);
	int niters = 1;
	dilate(con, con, Mat(), Point(-1, -1), niters);
	erode(con, con, Mat(), Point(-1, -1), niters*2);
	dilate(con, con, Mat(), Point(-1, -1), niters);
    findContours(con, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	vector<Point> poly; 
	vector<Rect> polyRect;
	int idx = 0;
	Mat result;
	tmp.copyTo(result);
	for(; idx >= 0; idx = hierarchy[idx][0])
	{
		approxPolyDP(Mat(contours[idx]), poly, 0.001, true);
		vector<Point>::const_iterator itp = poly.begin(); 

		while (itp != (poly.end() - 1)) 
		{ 
			line(result, *itp, *(itp + 1), Scalar(255), 1); 
			++itp; 
		} 
		line(result, *itp, *(poly.begin()), Scalar(255), 1);
		//imshow("imageWithContour", result);
		//waitKey(0);
	}
	//Mat a;
	//image.copyTo(a);
	int dis = 10;
	//cout << poly.size() << endl;
	for(int i = 0; i < poly.size(); i = i + 5)
	{
		Point lu, rd;
		lu.x = max(poly[i].x - dis, 0);
		lu.y = max(poly[i].y - dis, 0);
		rd.x = min(poly[i].x + dis, tmp.cols - 1);
		rd.y = min(poly[i].y + dis, tmp.rows - 1);
		Rect r(lu, rd);
		polyRect.push_back(r);
		rectangle( result,lu, rd, YELLOW, 1);
		//imshow( "��һ֡��ʼ������", result);
		//waitKey(0);
	}
	//waitKey(0);

	PolygonF pyF;
	for(int i = 0; i < poly.size(); i = i + 1)
	{

		PointF pt;
		pt.x = poly[i].x;
		pt.y = poly[i].y;
		pyF.mPoints.push_back(pt);
		
	}
	AddForegroundBorder(&pyF);
	mVideoSnapCut->SetForegroundBorderPolygons(&mForegroundBorderPolygons);//�ӻ����л�õ����������
	mVideoSnapCut->CreateMasks();
	mVideoSnapCut->BuildColorClassifiers();
	mVideoSnapCut->ClassifyPixels();
	mVideoSnapCut->ResampleContour();
	((CApp*)AfxGetApp())->mRightView->Invalidate();
	//AddForegroundBorder(&pyF);
	/*mForegroundBorderPolygons.push_back(&pyF);
	mVideoSnapCut->mForegroundBorderPolygons = &mForegroundBorderPolygons;*/
	cvDestroyWindow( winName.c_str() );
	cvDestroyWindow("�ָ���");
}
