#include "../../include/ImageFile/LBPfeatures.h"

//˫���Բ�ֵ==============================================================================
void  BilinearInterpolation(const Mat& img, float x, float y, int& value)
{
	int x_floor = floor(x);
	int x_ceil = ceil(x);
	int y_floor = floor(y);
	int y_ceil = ceil(y);
	uchar f00 = img.at<uchar>(y_floor, x_floor);
	uchar f10 = img.at<uchar>(y_floor, x_ceil);
	uchar f01 = img.at<uchar>(y_ceil, x_floor);
	uchar f11 = img.at<uchar>(y_ceil, x_ceil);
	value = ((x - x_floor) * f00 + (x_ceil - x) * f10) * (y - y_floor)+
		((x - x_floor) * f01 + (x_ceil - x) * f11) * (y_ceil - y);
}
//========================================================================================

//ͳ��������==============================================================================
int ComputeJumpNum(vector<bool>& res)
{
	int jumpNum = 0;
	for (int i = 0; i < res.size(); ++i)
	{
		if (i < res.size() - 1 && res[i] != res[i + 1])
			jumpNum++;
		if ((i == res.size() - 1) && res[i] != res[0])
			jumpNum++;
	}
	return jumpNum;
}
//========================================================================================

//��ȡLBP����=============================================================================
void ExtractLBPFeature(const Mat& srcImg, Mat& lbpFeature, float raduis, int ptsNum)
{
	if (lbpFeature.empty())
		lbpFeature = Mat(srcImg.size(), CV_8UC1, cv::Scalar(0));
	else if (lbpFeature.size() != lbpFeature.size())
	{
		lbpFeature.release();
		lbpFeature = Mat(srcImg.size(), CV_8UC1, cv::Scalar(0));
	}
	const uchar* pSrcImg = srcImg.ptr<uchar>();
	uchar* pLBPImg = lbpFeature.ptr<uchar>();
	int r = srcImg.rows;
	int c = srcImg.cols;
	float angleStep = CV_2PI / (float)ptsNum;
	for (int y = 0; y < r; ++y)
	{
		int offset = y * c;
		for (int x = 0; x < c; ++x)
		{
			int idx = -1;
			for (float angle = 0; angle < CV_2PI; angle += angleStep)
			{
				idx++;
				float x_ = x + raduis * std::cos(angle);
				float y_ = y + raduis * std::sin(angle);
				if (x_ < EPS || y_ < EPS || x_  > c- 1 || y_ > r-1)
					continue;
				int value = 0;
				BilinearInterpolation(srcImg, x_, y_, value);
				pLBPImg[offset + x] += (value < (int)pSrcImg[offset + x] ? 0 : 1 << idx);
			}
		}
	}
}
//========================================================================================

//LBP���ֱ��=============================================================================
void LBPDetectLine(const Mat& srcImg, Mat& lbpFeature, float raduis, int ptsNum)
{
	if (lbpFeature.empty())
		lbpFeature = Mat(srcImg.size(), CV_8UC1, cv::Scalar(0));
	else if (lbpFeature.size() != lbpFeature.size())
	{
		lbpFeature.release();
		lbpFeature = Mat(srcImg.size(), CV_8UC1, cv::Scalar(0));
	}
	const uchar* pSrcImg = srcImg.ptr<uchar>();
	uchar* pLBPImg = lbpFeature.ptr<uchar>();
	int r = srcImg.rows;
	int c = srcImg.cols;
	float angleStep = CV_2PI / (float)ptsNum;
	for (int y = 0; y < r; ++y)
	{
		int offset = y * c;
		for (int x = 0; x < c; ++x)
		{
			int idx = -1;
			vector<bool> label_(ptsNum, false);
			int numBig = 0, numSmall = 0;
			for (float angle = 0; angle < CV_2PI; angle += angleStep)
			{
				idx++;
				float x_ = x + raduis * std::cos(angle);
				float y_ = y + raduis * std::sin(angle);
				if (x_ > EPS && y_ > EPS && x_ < c - 1 && y_ < r - 1)
				{
					int value = 0;
					BilinearInterpolation(srcImg, x_, y_, value);
					if (value > (int)pSrcImg[offset + x])
					{
						label_[idx] = true;
						++numBig;
					}
					else
					{
						++numSmall;
					}
				}
			}
			int jumpNum = ComputeJumpNum(label_);

			//ֱ��
			//if (jumpNum == 2 && (numBig >= numSmall - 1 || numBig <= numSmall + 1))
			//	pLBPImg[offset + x] = 255;
			//�ǵ�
			if (jumpNum == 2 && (numBig < ptsNum / 2 - 2 || numBig > ptsNum / 2 + 2))
				pLBPImg[offset + x] = 255;
			//԰���߹�����
			if(numBig == ptsNum || numSmall == ptsNum)
				pLBPImg[offset + x] = 255;
		}
	}
}
//========================================================================================

void LBPfeaturesTest()
{
	cv::Mat image = cv::imread("Rect.bmp", 0);
	cv::Mat lbpFeature;
	LBPDetectLine(image, lbpFeature, 2, 8);
}

void TestMMSER()
{
	cv::Mat image = cv::imread("C:/Users/Administrator/Desktop/2.jpg", 0);

	cv::Ptr<cv::MSER> ptrMSER = cv::MSER::create(5, 800000, 1000000, 0.5);

	std::vector<std::vector<cv::Point> > points;
	std::vector<cv::Rect> rects;
	ptrMSER->detectRegions(image, points, rects);

	cv::Mat output(image.size(), CV_8UC3);
	output = cv::Scalar(255, 255, 255);
	cv::RNG rng;
	// ���ÿ����⵽�����������ڲ�ɫ������ʾ MSER
	// ������������ʾ�ϴ�� MSER
	for (std::vector<std::vector<cv::Point> >::reverse_iterator
		it = points.rbegin();
		it != points.rend(); ++it) {
		// ���������ɫ
		cv::Vec3b c(rng.uniform(0, 254),
			rng.uniform(0, 254), rng.uniform(0, 254));
		// ��� MSER �����е�ÿ����
		for (std::vector<cv::Point>::iterator itPts = it->begin();
			itPts != it->end(); ++itPts) {
			// ����д MSER ������
			if (output.at<cv::Vec3b>(*itPts)[0] == 255) {
				output.at<cv::Vec3b>(*itPts) = c;
			}
		}
	}

	cv::imshow("image", image);
	cv::imshow("output", output);
	waitKey(0);
}