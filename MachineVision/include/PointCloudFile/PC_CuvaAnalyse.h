#pragma once
#include "../BaseOprFile/utils.h"

const float CUVATHRES = 1e-6;

enum SURFTYPE
{
	PLANE = 0,
	RIDGE = 1,
	VALLEY = 2,
	SADDLERIDGE = 3,
	SADDLEVALLEY = 4,
	PEAK = 5,
	TRAP = 6,
	MINPOINT = 7
};

//拟合二次多项式
void FitQuadPoly(PC_XYZ& srcPC, Eigen::MatrixXf& res);

//求解极值点
void ComputeExtPt(Eigen::MatrixXf& conf, P_XYZ& extPt);

//曲率分类
SURFTYPE CuvaClass(Eigen::MatrixXf& conf);

//曲率分析
void CuvaAnalyse(PC_XYZ& srcPC, PC_XYZ& dstPC);

//计算特征点权重
float ComputeFPtW(const vector<float>& Ks, const vector<int>& PIdx);

//平均曲率
void MeanCuraAnalyse(PC_XYZ& srcPC, PC_XYZ& dstPC, int k_n, float thresW);

//曲率分析测试
void CuvaAnalyseTest();
