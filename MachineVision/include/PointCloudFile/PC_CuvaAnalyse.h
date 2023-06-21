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

//��϶��ζ���ʽ
void FitQuadPoly(PC_XYZ& srcPC, Eigen::MatrixXf& res);

//��⼫ֵ��
void ComputeExtPt(Eigen::MatrixXf& conf, P_XYZ& extPt);

//���ʷ���
SURFTYPE CuvaClass(Eigen::MatrixXf& conf);

//���ʷ���
void CuvaAnalyse(PC_XYZ& pc, PC_XYZ& dstPC);

//���ʷ�������
void CuvaAnalyseTest();
