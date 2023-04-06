#pragma once
#include "CPDMatch.h"

class RigidCPD :public CPD
{
public:
	RigidCPD(float w = 0.2f, int maxIter = 50, float eps = 1e-6f)
		:CPD(w, maxIter, eps)
	{}

	//��׼
	void Match(PC_XYZ &XPC, PC_XYZ &YPC);

	void GetResMat(Eigen::MatrixXf &resMat);

	float GetMagnification()
	{
		return m_S;
	}

private:
	//��ʼ�����Ա任����
	void InitRigidCompute(PC_XYZ &XPC, PC_XYZ &YPC);

	//����XTip, YTip
	void ComputeXTipYTip();

	//������Ա任����
	void ComputeRigidTranMat();

	//�������任����
	void ComputeAffineTranMat();

	//���Ƹ��Ա任
	void RigidTransPC();

	//���Ա任��ʹ��
	float m_S;   //�Ŵ���
	Eigen::Matrix3f m_RotMat;  //��ת����
	Eigen::MatrixXf m_TranMat; //ƽ�ƾ���
	Eigen::MatrixXf m_XTipMat;
	Eigen::MatrixXf m_YTipMat;
	Eigen::MatrixXf m_Ux;
	Eigen::MatrixXf m_Uy;
};

void TestRigidMatch();
