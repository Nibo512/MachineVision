#pragma once
#include "CPDMatch.h"
#include "../../../include/PointCloudMatch/CPDMatch/GPUCPD.h"

class NonRigidCPD :public CPD
{
public:
	NonRigidCPD(float w = 0.2f, int maxIter = 50, float eps = 1e-6f, bool isCpu = true) :
		CPD(w, maxIter, eps), m_IsCpu(isCpu)
	{

	}

	~NonRigidCPD();

	//非刚性配准
	void Match(PC_XYZ &XPC, PC_XYZ &YPC);

	void GetResMat(Eigen::MatrixXf &resMat);

private:
	//初始化非刚性变换计算
	void InitNonRigidCompute(PC_XYZ &XPC, PC_XYZ &YPC);

	//构造G矩阵
	void ConstructGMat();

	//计算非刚性变换矩阵
	void ComputeNonRigidTranMat();

	//计算低阶矩阵计算A
	void ComputeA(Eigen::MatrixXf &A, Eigen::MatrixXf &P1, float c);

	//点云非刚性变换
	void NonRigidTransPC();

	//GPU计算A
	void GPUComputeA(Eigen::MatrixXf& A, Eigen::MatrixXf& P1, float c);

private:
	Eigen::MatrixXf m_GMat;
	Eigen::MatrixXf m_WMat;
	Eigen::MatrixXf m_GDig;
	Eigen::MatrixXf m_GQ;
	float m_Beta;
	float m_Lamda;
	int m_LowRankN;

	bool m_IsCpu;

	//GPU部分
	float* m_pD_A;
	float* m_pD_C;

	cublasHandle_t m_CublasHandle;

	cusolverDnHandle_t m_CusolverHandle;
};

void TestNonRigidMatch();