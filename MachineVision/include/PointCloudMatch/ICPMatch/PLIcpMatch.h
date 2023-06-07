#pragma once
#include "IcpMatch.h"

class PLICP : public ICP
{
public:
	PLICP():ICP()
	{
	};

public:
	//匹配程序
	float Match(PC_XYZ &srcPC, PC_XYZ &tgtPC, Eigen::Matrix4f &transMat);

private:
	//计算变换矩阵---仿射变换
	void ComputeAffineTransMat(Eigen::Matrix4f &transMat);

	//刚性变换
	void ComputeRigidTranMat(Eigen::Matrix4f &transMat);

	//带法向量的
	void ComputeTransMatWithNormal(Eigen::Matrix4f &transMat);

	//计算loss
	bool ComputeLoss();

	//计算匹配分数
	float ComputeFitScore();

private:
	PC_N m_TgtPtVecs;
};

void PLIcpMatchTest();
