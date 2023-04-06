#pragma once
#include "FFT.h"
#include "../BaseOprFile/OpenCV_Utils.h"

/*˵����
	Img����ͷ��ʾ�����˲�
	ImgF����ͷ��ʾƵ�����˲�
	srcImg��[in]���˲�ͼ��
	dstImg��[out]�˲����ͼ��
*/

/*�����˲�:
	guidImg��[in]����ͼ��	
	size��[in]�˲�����С
*/
void Img_GuidFilter(Mat &srcImg, Mat &guidImg, Mat &dstImg, int size, float eps);

/*����ӦCanny�˲�
	size��[in]�˲�����С
	sigma��[in]�ߵ���ֵ����
*/
void Img_AdaptiveCannyFilter(Mat &srcImg, Mat &dstImg, int size, double sigma);

/*Ƶ�����˲�
	srcImg����ͨ��ͼ��
	lr��[in]�˲��Ͱ뾶
	hr��[in]�˲��߰뾶
	�õ�ͨ�˲���ʱȡlr����״�˲������߶�ȡ����Ϊ BLPF �˲���ʱhrΪָ�� n
	passMode��[in]��ʾ��ͨ���߸�ͨ--0 ��ʾ��ͨ��1 ��ʾ��ͨ
	filterMode��[in]��ʾ�˲�������
*/
void ImgF_FreqFilter(Mat &srcImg, Mat &dstImg, double lr, double hr, int passMode, IMGF_MODE filterMode);

/*̩ͬ�˲���
	radius��[in]�˲��뾶
	L��[in]�ͷ���
	H��[in]�߷���
	c��[in]ָ������ϵ��
*/
void ImgF_HomoFilter(Mat &srcImg, Mat &dstImg, double radius, double L, double H, double c);

/*��������ƽ����
	lamda��[in]����ƽ���̶�
	step_t��[in]ʱ�䲽��
	iter_k��[in]��������
*/
void Img_AnisotropicFilter(Mat &srcImg, Mat &dstImg, double lamda, double step_t, int iter_k);

/*��˹�˲�:
	h���˲�����������
	w���˲����ĺ�����
*/
void Img_GaussFilter(Mat &srcImg, Mat &dstImg, int h, int w);

void FilterTest();
