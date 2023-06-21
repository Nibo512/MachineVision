#pragma once
#include "../BaseOprFile/utils.h"

//获取参数T
void GetCurveTParam(vector<vector<float>>& t, int size, int order, float min_t, float max_t);

//获取参数矩阵A
void CalCurveMatA(Eigen::MatrixXf& matA, vector<vector<float>>& t);

//获取参数矩阵B
void CalCurveMatB(Eigen::MatrixXf& matB, vector<vector<float>>& t, vector<float>& x);

//拟合多项式
void FitCurvePoly(vector<float>& x, vector<vector<float>>& t, vector<float>& coff, int order);

//多项式平滑
void PolyCurveSmooth(PC_XYZ& srcPC, PC_XYZ& dstPC, int size, int order, float min_t, float max_t);

void PolyCurveFitmoothTest();