#pragma once
#include "../BaseOprFile/OpenCV_Utils.h"
#include "opencv2/features2d.hpp"

//˫���Բ�ֵ
void BilinearInterpolation(const Mat& img, float x, float y, int& value);

//ͳ��������
int ComputeJumpNum(vector<bool>& res);

//��ȡLBP����
void ExtractLBPFeature(const Mat& srcImg, Mat& lbpFeature, float raduis, int ptsNum);

//LBP���ֱ��
void LBPDetectLine(const Mat& srcImg, Mat& lbpFeature, float raduis, int ptsNum);

void LBPfeaturesTest();

void TestMMSER();
