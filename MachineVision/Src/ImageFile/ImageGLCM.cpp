#include "../../include/ImageFile/ImageGLCM.h"

//ͼ��ĻҶȹ���������============================================================
void Img_ComputeGLCM(Mat &srcImg, Mat &MatGLCM, int grayLevel, GLCM_ORIT angle)
{
	CV_CheckGE(grayLevel, 0, "�Ҷ�ֵ����С��0");
	CV_CheckLE(grayLevel, 255, "�Ҷ�ֵ���ܴ���255");
	Mat image = Mat(srcImg.size(), CV_8UC1);
	MatGLCM = Mat(grayLevel + 1, grayLevel + 1, CV_16SC1, Scalar::all(0));
	for (int y = 0; y < image.rows; y++)
	{
		uchar *pSrcData = srcImg.ptr<uchar>(y);
		uchar *pImgData = image.ptr<uchar>(y);
		for (int x = 0; x < image.cols; x++)
		{
			pImgData[x] = (uchar)pSrcData[x] * grayLevel / 255;
		}
	}
	switch (angle)
	{
	case GLCM_HORIZATION:
		Img_GLCM_0(image, MatGLCM);
		break;
	case GLCM_ANGLE45:
		Img_GLCM_45(image, MatGLCM);
		break;
	case GLCM_VERTICAL:
		Img_GLCM_90(image, MatGLCM);
		break;
	case GLCM_ANGLE135:
		Img_GLCM_135(image, MatGLCM);
		break;
	default:
		break;
	}
}
//================================================================================

//0�ȻҶȹ�����===================================================================
void Img_GLCM_0(Mat& srcImg, Mat& MatGLCM)
{
	int r = srcImg.rows;
	int c = srcImg.cols;
	ushort *pMatGLCM = MatGLCM.ptr<ushort>(0);
	int grayLevel = MatGLCM.rows;
	for (int y = 0; y < r; y++)
	{
		uchar *pData = srcImg.ptr<uchar>(y);
		for (int x = 0; x < c - 1; x++)
		{
			(*(pMatGLCM + grayLevel * pData[x] + pData[x + 1])) += 1;
		}
	}
}
//================================================================================

//45�ȻҶȹ�����==================================================================
void Img_GLCM_45(Mat& srcImg, Mat& MatGLCM)
{
	int r = srcImg.rows;
	int c = srcImg.cols;
	ushort *pMatGLCM = MatGLCM.ptr<ushort>(0);
	int grayLevel = MatGLCM.rows;
	for (int y = 0; y < r - 1; y++)
	{
		uchar *pDataUp = srcImg.ptr<uchar>(y);
		uchar *pDataDown = srcImg.ptr<uchar>(y + 1);
		for (int x = 0; x < c - 1; x++)
		{
			(*(pMatGLCM + grayLevel * pDataUp[x] + pDataDown[x + 1])) += 1;
		}
	}
}
//================================================================================

//90�ȻҶȹ�����==================================================================
void Img_GLCM_90(Mat& srcImg, Mat& MatGLCM)
{
	int r = srcImg.rows;
	int c = srcImg.cols;
	ushort *pMatGLCM = MatGLCM.ptr<ushort>(0);
	int grayLevel = MatGLCM.rows;
	for (int y = 0; y < r - 1; y++)
	{
		uchar *pDataUp = srcImg.ptr<uchar>(y);
		uchar *pDataDown = srcImg.ptr<uchar>(y + 1);
		for (int x = 0; x < c; x++)
		{
			(*(pMatGLCM + grayLevel * pDataUp[x] + pDataDown[x])) += 1;
		}
	}
}
//================================================================================

//135�ȻҶȹ�����=================================================================
void Img_GLCM_135(Mat& srcImg, Mat& MatGLCM)
{
	int r = srcImg.rows;
	int c = srcImg.cols;
	ushort *pMatGLCM = MatGLCM.ptr<ushort>(0);
	int grayLevel = MatGLCM.rows;
	for (int y = 0; y < r - 1; y++)
	{
		uchar *pDataUp = srcImg.ptr<uchar>(y);
		uchar *pDataDown = srcImg.ptr<uchar>(y + 1);
		for (int x = 1; x < c; x++)
		{
			(*(pMatGLCM + grayLevel * pDataUp[x] + pDataDown[x - 1])) += 1;
		}
	}
}
//================================================================================

//������Ƶ��======================================================================
void cal_total_number(Mat &MatGLCM, int &number)
{
	int r = MatGLCM.rows;
	int c = MatGLCM.cols;
	if (MatGLCM.isContinuous())
	{
		c *= r;
		r = 1;
	}
	number = 0;
	uchar *pMatGLCM = MatGLCM.data;
	for (int i = 0; i < r; ++i, pMatGLCM += c)
	{
		for (int j = 0; j < c; ++j)
		{
			number += (int)pMatGLCM[j];
		}
	}
}
//================================================================================

//����Ҷȹ����ص�����/һ����=====================================================
void Img_GLCMEnergy(Mat& MatGLCM, double& energy)
{
	int r = MatGLCM.rows;
	int c = MatGLCM.cols;
	energy = 0;
	float *pData = MatGLCM.ptr<float>(0);
	for (int y = 0; y < r; pData += c, ++y)
	{
		for (int x = 0; x < c; ++x)
		{
			energy += (pData[x] * pData[x]);
		}
	}
}
//================================================================================

//��ûҶȹ����ص�����ͬ����======================================================
void Img_GLCMHomogeneity(Mat& MatGLCM, double& homogeneity)
{
	int r = MatGLCM.rows;
	int c = MatGLCM.cols;
	homogeneity = 0;
	float *pData = MatGLCM.ptr<float>(0);
	for (int y = 0; y < r; pData += c, ++y)
	{
		for (int x = 0; x < c; ++x)
		{
			homogeneity += (pData[x] / (1 + abs(x - y)));
		}
	}
}
//================================================================================

//��û�������صĶԱȶ�==========================================================
void Img_GLCMContrast(Mat& MatGLCM, double& contrast)
{
	int r = MatGLCM.rows;
	int c = MatGLCM.cols;
	contrast = 0;
	float *pData = MatGLCM.ptr<float>(0);
	for (int y = 0; y < r; pData += c, ++y)
	{
		for (int x = 0; x < c; ++x)
		{
			contrast += (pData[x] * (x - y) * (x - y));
		}
	}
}
//================================================================================

//��ûҶȹ����ص���==============================================================
void Img_GLCMEntropy(Mat& MatGLCM, double& entropy)
{
	int r = MatGLCM.rows;
	int c = MatGLCM.cols;
	entropy = 0;
	float *pData = MatGLCM.ptr<float>(0);
	for (int y = 0; y < r; pData += c, ++y)
	{
		for (int x = 0; x < c; ++x)
		{
			if (pData[x] != 0)
			{
				entropy -= (pData[x] * log2(pData[2]));
			}
		}
	}
}
//================================================================================

//��ûҶȹ����������ض�========================================================
void Img_GLCMCorrelation(Mat& MatGLCM, double& correlation)
{
	int r = MatGLCM.rows;
	int c = MatGLCM.cols;
	correlation = 0;
	float ux = 0;
	float uy = 0;
	float sx = 0;
	float sy = 0;
	get_ux_uy(MatGLCM, ux, uy);
	get_sx_sy(MatGLCM, ux, uy, sx, sy);
	float *pData = MatGLCM.ptr<float>(0);
	for (int y = 0; y < r; pData += c, ++y)
	{
		float correlation_1 = 0;
		for (int x = 0; x < c; ++x)
		{
			correlation_1 += (pData[x] * (x - ux));
		}
		correlation += (correlation_1 * (y - uy));
	}
	correlation = correlation / sx / sy;
}
//================================================================================

//���ux��uy
void get_ux_uy(Mat &MatGLCM, float &ux, float &uy)
{
	int r = MatGLCM.rows;
	int c = MatGLCM.cols;
	ux = 0;
	uy = 0;
	float *pData = MatGLCM.ptr<float>(0);
	for (int y = 0; y < r; pData += c, ++y)
	{
		float uy_1 = 0;
		for (int x = 0; x < c; ++x)
		{
			ux += (pData[x] * x);
			uy_1 += pData[x];
		}
		uy += (uy_1 * y);
	}
}

//���sx��sy
void get_sx_sy(Mat &MatGLCM, float &ux, float &uy, float &sx, float &sy)
{
	int r = MatGLCM.rows;
	int c = MatGLCM.cols;
	sx = 0;
	sy = 0;
	float *pData = MatGLCM.ptr<float>(0);
	for (int y = 0; y < r; pData += c, ++y)
	{
		float diff_y = (y - uy);
		float sy_1 = 0;
		for (int x = 0; x < c; ++x)
		{
			float diff_x = (x - ux);
			sx += (pData[x] * diff_x * diff_x);
			sy_1 += pData[x];
		}
		sy += (sy_1 * diff_y*diff_y);
	}
}