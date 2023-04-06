#pragma once
#include "../BaseOprFile/utils.h"

class SurfMatch
{
	struct KeyPoint
	{
		float x;
		float y;
		int scale;
		float amplitude;
		bool operator<(const KeyPoint &a) const
		{
			return this->amplitude > a.amplitude;
		}
	};

public:
	SurfMatch(const Mat &srcImg, double ampThres = 0.0);

//private:
	//计算偏移
	void ComputeOffset(const int pSrc[][5], int size, int minSize, int num, int p_[][5]);

	//方框滤波
	void BoxFilter(const int px[][5], const int py[][5], const int pxy[][5], int size, Mat &image);

	//计算金字塔特征图
	void ComputePyrMaps();

	//计算方框滤波大小
	void ComputeBoxFilterSizes(vector<int> &sizes, int start_size);

	//非极大值抑制
	void NMSPts(Mat &img1, Mat &img2, Mat &img3, int size);

	//提取极值点
	void GetExtremumPts();

	//计算特征点的主方向
	void ComputeaKeyPtMajorOri();


//private:
	vector<Mat> m_PyrImgs;

	//积分图
	Mat m_IntImg;

	int m_nOctaves;
	int m_nOctaveLayers;
	//Mat m_SrcImg;

	int m_ImgW;
	int m_ImgH;
	double m_AmpThres;

	vector<KeyPoint> m_KeyPts;

	const int m_Offset_x[3][5] = { {0, 2, 3, 7, 1}, {3, 2, 6, 7, -2}, {6, 2, 9, 7, 1} };
	const int m_Offset_y[3][5] = { {2, 0, 7, 3, 1}, {2, 3, 7, 6, -2}, {2, 6, 7, 9, 1} };
	const int m_Offset_xy[4][5] = { {1, 1, 4, 4, 1}, {5, 1, 8, 4, -1}, {1, 5, 4, 8, -1}, {5, 5, 8, 8, 1} };

	Mat m_Mask;

	vector<int> m_Sizes;

	int m_PtNum;
};

void SurfMatchTest();