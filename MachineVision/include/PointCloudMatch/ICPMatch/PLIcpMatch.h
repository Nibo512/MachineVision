#pragma once
#include "IcpMatch.h"

class PLICP : public ICP
{
public:
	PLICP():ICP()
	{
	};

public:
	//ƥ�����
	float Match(PC_XYZ &srcPC, PC_XYZ &tgtPC, Eigen::Matrix4f &transMat);

private:
	//����任����---����任
	void ComputeAffineTransMat(Eigen::Matrix4f &transMat);

	//���Ա任
	void ComputeRigidTranMat(Eigen::Matrix4f &transMat);

	//����������
	void ComputeTransMatWithNormal(Eigen::Matrix4f &transMat);

	//����loss
	bool ComputeLoss();

	//����ƥ�����
	float ComputeFitScore();

private:
	PC_N m_TgtPtVecs;
};

void PLIcpMatchTest();
