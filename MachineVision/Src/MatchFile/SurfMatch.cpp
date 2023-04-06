#include "../../include/MatchFile/SurfMatch.h"

SurfMatch::SurfMatch(const Mat &srcImg, double ampThres)
{
	cv::Mat grayImg;
	if (srcImg.channels() == 3)
	{
		cvtColor(srcImg, grayImg, COLOR_BGR2GRAY);
	}
	else if (srcImg.channels() == 1)
	{
		grayImg = srcImg.clone();
	}
	else
		return;
	m_ImgW = grayImg.cols;
	m_ImgH = grayImg.rows;
	//计算积分图
	cv::integral(grayImg, m_IntImg, CV_32SC1);
	m_AmpThres = ampThres;

	m_nOctaves = 4;
	m_nOctaveLayers = 4;

	m_PtNum = 100;
}

//计算Dx、Dy、Dxy
float CompupteDxDyDxy(int *pIntImg, const int p_[][5], int num)
{
	float sum = 0.0f;
	for (int i = 0; i < num; ++i)
	{
		sum += (pIntImg[p_[i][0]] + pIntImg[p_[i][3]] - pIntImg[p_[i][1]] - pIntImg[p_[i][2]]) * p_[i][4];
	}
	return sum;
}

//方框滤波
void SurfMatch::BoxFilter(const int px[][5], const int py[][5], const int pxy[][5], int size, Mat &image)
{
	image = Mat(cv::Size(m_ImgW, m_ImgH), CV_32FC1, cv::Scalar::all(0));
	if (size > m_ImgH - 1 || size > m_ImgW - 1)
		return;
	int halfSize = size / 2;
	int end_x = m_ImgW - halfSize;
	int end_y = m_ImgH - halfSize;
	for (int y = halfSize; y < end_y; ++y)
	{
		int *pImg = m_IntImg.ptr<int>(y - halfSize);
		float *pImage = image.ptr<float>(y);
		for (int x = halfSize; x < end_x; ++x)
		{
			int xIdx = x - halfSize;
			int* pIntImg = pImg + xIdx;
			float dxx = CompupteDxDyDxy(pIntImg, px, 3);
			float dyy = CompupteDxDyDxy(pIntImg, py, 3);
			float dxy = CompupteDxDyDxy(pIntImg, pxy, 4);
			pImage[x] = (dxx * dyy - 0.81f * dxy * dxy);
		}
	}
}

//计算偏移
void SurfMatch::ComputeOffset(const int pSrc[][5], int size, int minSize, int num, int pDst[][5])
{
	int width = m_IntImg.cols;
	float ratio = (float)size / float(minSize);
	for (int i = 0; i < num; ++i)
	{
		int dx1 = cvRound(ratio*pSrc[i][0]);
		int dy1 = cvRound(ratio*pSrc[i][1]);
		int dx2 = cvRound(ratio*pSrc[i][2]);
		int dy2 = cvRound(ratio*pSrc[i][3]);

		pDst[i][0] = dy1 * width + dx1;
		pDst[i][1] = dy2 * width + dx1;
		pDst[i][2] = dy1 * width + dx2;
		pDst[i][3] = dy2 * width + dx2;
		pDst[i][4] = pSrc[i][4];
	}
}

//计算金字塔特征图
void SurfMatch::ComputePyrMaps()
{
	int nTotalNum = m_nOctaves * m_nOctaveLayers;
	m_Sizes.resize(nTotalNum);
	ComputeBoxFilterSizes(m_Sizes, 9);
	m_PyrImgs.resize(nTotalNum);
	for (int i = 0; i < nTotalNum; ++i)
	{
		int px[3][5];
		int py[3][5];
		int pxy[4][5];
		ComputeOffset(m_Offset_x, m_Sizes[i], 9, 3, px);
		ComputeOffset(m_Offset_y, m_Sizes[i], 9, 3, py);
		ComputeOffset(m_Offset_xy, m_Sizes[i], 9, 4, pxy);
		BoxFilter(px, py, pxy, m_Sizes[i], m_PyrImgs[i]);
		m_PyrImgs[i] /= ((float)m_Sizes[i] * (float)m_Sizes[i]);
	}
}

//计算方框滤波大小
void SurfMatch::ComputeBoxFilterSizes(vector<int> &sizes, int start_size)
{
	int index = 0;
	for (int i = 0; i < m_nOctaves; ++i)
	{
		int step = 6 * pow(2, i);
		int start_size_ = start_size;
		for (int j = 0; j < i; ++j)
		{
			start_size_ += 6 * pow(2, j);
		}
		for (int j = 0; j < m_nOctaveLayers; ++j)
		{
			sizes[index] = start_size_ + step * j;
			++index;
		}
	}

	//for (int octave = 0; octave < m_nOctaves; octave++)
	//{
	//	for (int layer = 0; layer < m_nOctaveLayers; layer++)
	//	{
	//		sizes[index] = (9 + 6 * layer) << octave;
	//		cout << sizes[index] << ",";
	//		index++;
	//	}
	//	cout << endl;
	//}
}

//非极大值抑制
void SurfMatch::NMSPts(Mat &prevImg, Mat &curImg, Mat &nextImg, int size)
{
	float *pPrev = prevImg.ptr<float>(1);
	float *pCur = curImg.ptr<float>(1);
	float *pNext = nextImg.ptr<float>(1);
	uchar *pMask = m_Mask.ptr<uchar>(1);

	//边界处不计算
	for (int y = 1; y < m_ImgH - 1; ++y, pMask += m_ImgW,
		pCur += m_ImgW, pPrev += m_ImgW, pNext += m_ImgW)
	{
		for (int x = 1; x < m_ImgW - 1; ++x)
		{
			float curVal = (pCur[x]);
			if (curVal > m_AmpThres && pMask[x] == 0)
			{
				int res = 1;
				float Neibour[27] = { (pPrev - m_ImgW)[x - 1],(pPrev - m_ImgW)[x], (pPrev - m_ImgW)[x + 1],
									   pPrev[x - 1],           pPrev[x],            pPrev[x + 1],
									   (pPrev + m_ImgW)[x - 1],(pPrev + m_ImgW)[x], (pPrev - m_ImgW)[x + 1],

									   (pCur - m_ImgW)[x - 1],(pCur - m_ImgW)[x], (pCur - m_ImgW)[x + 1],
									   pCur[x - 1],          m_AmpThres,            pCur[x + 1],
									   (pCur + m_ImgW)[x - 1],(pCur + m_ImgW)[x], (pCur - m_ImgW)[x + 1] ,

									   (pNext - m_ImgW)[x - 1],(pNext - m_ImgW)[x], (pNext - m_ImgW)[x + 1],
								       pNext[x - 1],          pNext[x],            pNext[x + 1],
									   (pNext + m_ImgW)[x - 1],(pNext + m_ImgW)[x], (pNext - m_ImgW)[x + 1] };
				for (int i = 0; i < 27; ++i)
				{
					if (curVal <= Neibour[i])
					{
						res = 0; break;
					}
				}
				if (res == 1)
				{
					KeyPoint kpt;
					kpt.x = x; kpt.y = y; kpt.amplitude = curVal; 
					kpt.scale = int(1.2f * size / 9.0f);
					m_KeyPts.push_back(kpt); pMask[x] = 255;
				}
			}
		}
	}
}

//提取极值点
void SurfMatch::GetExtremumPts()
{
	m_Mask = Mat(cv::Size(m_ImgW, m_ImgH), CV_8UC1, cv::Scalar(0));
	for (int i = 0; i < m_PyrImgs.size(); i += m_nOctaveLayers)
	{
		for (int j = i + 1; j < i + m_nOctaveLayers - 1; ++j)
		{
			NMSPts(m_PyrImgs[j - 1], m_PyrImgs[j], m_PyrImgs[j + 1], m_Sizes[j]);
		}
	}
	std::stable_sort(m_KeyPts.begin(), m_KeyPts.end());
}

//计算特征点的主方向
void SurfMatch::ComputeaKeyPtMajorOri()
{
	m_PtNum = m_PtNum < m_KeyPts.size() ? m_PtNum : m_KeyPts.size();
	int width = m_IntImg.cols;
	for (int i = 0; i < m_PtNum; ++i)
	{
		int x = m_KeyPts[i].x;
		int y = m_KeyPts[i].y;
		int s_ = m_KeyPts[i].scale;
		int harrSize = s_ * 2;
		int ptIdxX[2][5] = { {0, harrSize * width, 2 * harrSize, harrSize * width + 2 * harrSize, -1},
							{harrSize * (width + 1),  2 * harrSize * (width + 1), 
							harrSize * (width + 1) + 2 * harrSize, 2 *  harrSize * (width + 1) + 2 * harrSize, 1} };
		int ptIdxY[2][5] = { {0, 2 * harrSize * width, harrSize, 2 * harrSize * width + harrSize, -1},
							{harrSize, harrSize + 2 * harrSize * width, 2 * harrSize, 2 * harrSize + 2 * harrSize * width, 1} };
		int range = s_ * 6;

		int start_y = y - range - harrSize > 0 ? y - range: harrSize;
		int end_y = y + range + harrSize < m_IntImg.rows ? y + range: m_IntImg.rows - harrSize;

		int start_x = x - range - harrSize > 0 ? x - range: harrSize;
		int end_x = x + range + harrSize < width ? x + range: width - harrSize;

		int size = (end_y - start_y) * (end_x - start_x);
		vector<float> dxx(size), dyy(size), mod(size), angle(size);
		int idx = 0;
		for (int y_ = start_y; y_ < end_y; y_ += s_)
		{
			for (int x_ = start_x; x_ < end_x; y += s_)
			{
				if (y_ * y_ + x_ * x_ < range * range)
				{
					dxx[idx] = CompupteDxDyDxy(&m_IntImg.at<int>(y, x), ptIdxX, 2);
					dyy[idx] = CompupteDxDyDxy(&m_IntImg.at<int>(y, x), ptIdxY, 2);
					mod[idx] = dxx[idx] * dxx[idx] + dyy[idx] * dyy[idx];
					angle[idx] = cv::fastAtan2(y_ - y, x_ - x);
					idx++;
				}
			}
		}
		float bestx = 0, besty = 0, bestMod = 0;
		for (int j = 0; j < 360; j += 12)
		{
			float sum_x = 0.0f, sum_y = 0.0f;
			for (int k = 0; k < size; ++k)
			{
				float diff_ang = angle[k] - j;
				if (diff_ang < 30 || diff_ang > 330)
				{
					sum_x += dxx[k]; sum_y += dyy[k];
				}
			}
			float mod_ = sum_x * sum_x + sum_y * sum_y;
			if (bestMod < mod_)
			{
				bestx = sum_x; besty = sum_y; bestMod = mod_;
			}
		}
	}
}

void SurfMatchTest()
{
	Mat image = imread("3.jpg", 1);
	SurfMatch match(image,500);
	Mat tImg = match.m_IntImg;
	match.ComputePyrMaps();
	match.GetExtremumPts();

	for (int i = 0; i < 50; ++i)
	{
		Point pt(match.m_KeyPts[i].x, match.m_KeyPts[i].y);
		cv::line(image, pt, pt, cv::Scalar(0, 0, 255), 5);
	}
}