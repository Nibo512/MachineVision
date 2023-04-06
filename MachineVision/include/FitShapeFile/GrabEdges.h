#pragma once
#include "../BaseOprFile/utils.h"

/*˵����
	srcImg��[in]ԭʼͼ��
	edges��[out]����ı߽�
*/

enum IMG_GRABEDGEMODE
{
	IMG_EDGE_LIGHT = 0,
	IMG_EDGE_DARK = 1,
	IMG_EDGE_ABSOLUTE = 2
};

/*��Բ���ķ�ʽץ�ߣ�
	center��[in]Բ������
	r_1��r_2��[in]СԲ������Բ���뾶 --- r_2 > r_1
	r_step��[in]ɨ�貽��
	startAng��endAng��[in]ɨ����ʼ������ֹ�Ƕ� --- startAng < endAng
	angStep��[in]�ǶȲ���
	thresVal��[in]��ֵ
	mode��[in]��Եģʽ---IMG_EDGE_LIGHT��ȡ���ߣ�IMG_EDGE_DARK��ȡ���ߣ�IMG_EDGE_ABSOLUTE����ȡ
	ptsNo��[in]Ҫѡ��ĵ����
	scanOrit��[in]ɨ�跽��----0����ʾ�ɰ뾶r_2��r_1��1����ʾ�뾶r_1��r_2
*/
void Img_GrabEdgesCircle(Mat& srcImg, vector<cv::Point>& edges, cv::Point& center, double r_1, double r_2, double r_step,
	double startAng, double endAng, double angStep, double thresVal, IMG_GRABEDGEMODE mode, int ptsNo, int scanOrit);

/*�Ծ��εķ�ʽץ��
	start_p��end_p��[in]��ʼ�㡢����ֹ��
	width��[in]ɨ����
	step1��[in]�㲽��
	step2��[in]ɨ�貽��
	thresVal��[in]��ֵ
	mode��[in]��Եģʽ---IMG_EDGE_LIGHT��ȡ���ߣ�IMG_EDGE_DARK��ȡ���ߣ�IMG_EDGE_ABSOLUTE����ȡ
	ptsNo��[in]Ҫѡ��ĵ����
	scanOrit��[in]ɨ�跽��
*/
void Img_GrabEdgesRect(Mat& srcImg, vector<cv::Point>& edges, cv::Point& start_p, cv::Point& end_p, int width,
	double step1, double step2, double thresVal, IMG_GRABEDGEMODE mode, int ptsNo, int scanOrit);