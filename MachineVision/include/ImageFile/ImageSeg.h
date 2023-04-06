#pragma once
#include "../BaseOprFile/OpenCV_Utils.h"

/*˵����
	srcImg��[in]����ǿͼ��
	dstImg��[out]��ǿ���ͼ��
*/

/*������ֵ�ָ�*/
void Img_Seg(Mat& srcImg, Mat& dstImg, double thres, IMG_SEG mode);

/*ѡ��Ҷ�����:
	thresVal1��[in]����ֵ
	thresVal2��[in]����ֵ
	mode��[in]��ֵ��ģʽ---IMG_SEG_LIGHT��gray > thresVal1 && gray < thresVal2
					   IMG_SEG_DARK��gray < thresVal1 && gray > thresVal2
*/
void Img_SelectGraySeg(Mat& srcImg, Mat& dstImg, uchar thresVal1, uchar thresVal2, IMG_SEG mode);

/*��������ֵ�ָ
	mode��[in]��ֵ��ģʽ---IMG_SEG_LIGHT��ѡ��ͼ�����Ĳ���
					   IMG_SEG_DARK��ѡ��ͼ�񰵵Ĳ���
*/
void Img_MaxEntropySeg(Mat& srcImg, Mat& dstImg, IMG_SEG mode);

/*��������Ӧ��ֵ��
	eps��[in]��ֹ����
	mode��[in]��ֵ��ģʽ---IMG_SEG_LIGHT��ѡ��ͼ�����Ĳ���
					IMG_SEG_DARK��ѡ��ͼ�񰵵Ĳ���
*/
void Img_IterTresholdSeg(Mat& srcImg, Mat& dstImg, double eps, IMG_SEG mode);

/*�ֲ�����Ӧ��ֵ�ָhalcon�е�var_threshold
	size��[in]�˲�����С
	stdDevScale��[in]��׼�������
	absThres��[in]������ֵ
	mode��[in]�˲�ģʽ
*/
void Img_LocAdapThresholdSeg(Mat& srcImg, Mat& dstImg, cv::Size size, double stdDevScale, double absThres, IMG_SEG mode);

/*���ͷָ
	thresVal1��[in]����ֵ
	thresVal2��[in]����ֵ
*/
void Img_HysteresisSeg(Mat& srcImg, Mat& dstImg, double thresVal1, double thresVal2);

/*����������
	dist_c��[in]ͼ���з���Ĳ���
	dist_r��[in]ͼ���з���Ĳ���
*/
void Img_RegionGrowSeg(Mat& srcImg, Mat& labels, int dist_c, int dist_r, int thresVal, int minRegionSize);


void ImgSegTest();
