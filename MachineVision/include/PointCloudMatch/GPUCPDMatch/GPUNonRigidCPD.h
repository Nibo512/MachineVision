#pragma once
#include "../../BaseOprFile/utils.h"
#include "../CPDMatch/CPDMatch.h"

class GPUNonRigidCPD : public CPD
{
public:
	GPUNonRigidCPD(float w = 0.2f, int maxIter = 50, float eps = 1e-6f) :
		CPD(w, maxIter, eps)
	{}

	~GPUNonRigidCPD();

public:
	//�Ǹ�����׼
	void Match(PC_XYZ& XPC, PC_XYZ& YPC);

	void GetResMat(Eigen::MatrixXf& resMat)
	{
		return;
	}

private:
	//��ʼ������
	void InitGPUCompute(PC_XYZ& XPC, PC_XYZ& YPC);

	//����P����
	void ComputetPMat();

	//����G����
	void ConstructGMat();

	//����ȥ���Ļ�
	void PCCentralization(PC_XYZ& XPC, PC_XYZ& YPC);

	////����A
	//void GPUComputeA(Eigen::MatrixXf& A, Eigen::MatrixXf& dPM, float c);

private:
	float m_Beta;
	float m_Lamda;

	float* m_pGMat_H;
	float* m_GDiagMat;

	PC_XYZ m_XPC;
	PC_XYZ m_YPC;
};

void TestGPUNonRigidMatch();