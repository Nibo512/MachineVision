#pragma once
#include "../BaseOprFile/utils.h"

const float CUVATHRES = 1e-3;

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
void CuvaAnalyse(PC_XYZ& pc, PC_XYZ& dstPC);

//曲率分析测试
void CuvaAnalyseTest();
