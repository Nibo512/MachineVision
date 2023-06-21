#pragma once
#include "../BaseOprFile/utils.h"

//��ȡ����T
void GetCurveTParam(vector<vector<float>>& t, int size, int order, float min_t, float max_t);

//��ȡ��������A
void CalCurveMatA(Eigen::MatrixXf& matA, vector<vector<float>>& t);

//��ȡ��������B
void CalCurveMatB(Eigen::MatrixXf& matB, vector<vector<float>>& t, vector<float>& x);

//��϶���ʽ
void FitCurvePoly(vector<float>& x, vector<vector<float>>& t, vector<float>& coff, int order);

//����ʽƽ��
void PolyCurveSmooth(PC_XYZ& srcPC, PC_XYZ& dstPC, int size, int order, float min_t, float max_t);

void PolyCurveFitmoothTest();