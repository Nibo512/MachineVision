#pragma once
#include "CPDMatch.h"

class NonRigidCPD :public CPD
{
public:
	NonRigidCPD(float w = 0.2f, int maxIter = 50, float eps = 1e-6f) :
		CPD(w, maxIter, eps)
	{

	}

	~NonRigidCPD();

	//�Ǹ�����׼
	void Match(PC_XYZ &XPC, PC_XYZ &YPC);

	void GetResMat(Eigen::MatrixXf &resMat);

private:
	//��ʼ���Ǹ��Ա任����
	void InitNonRigidCompute(PC_XYZ &XPC, PC_XYZ &YPC);

	//����G����
	void ConstructGMat();

	//����Ǹ��Ա任����
	void ComputeNonRigidTranMat();

	//����ͽ׾������A
	void ComputeA(Eigen::MatrixXf &A, Eigen::MatrixXf &P1, float c);

	//���ƷǸ��Ա任
	void NonRigidTransPC();

private:
	Eigen::MatrixXf m_GMat;
	Eigen::MatrixXf m_WMat;
	Eigen::MatrixXf m_GDig;
	Eigen::MatrixXf m_GQ;
	float m_Beta;
	float m_Lamda;
	int m_LowRankN;
};

void TestNonRigidMatch();