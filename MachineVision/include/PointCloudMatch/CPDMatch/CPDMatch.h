#pragma once

#include "../../BaseOprFile/utils.h"

class CPD
{
public:
	CPD(float w = 0.2f, int maxIter = 50, float eps = 1e-6f) :
		m_W(w), m_MaxIter(maxIter), m_Eps(eps)
	{}

public:
	virtual void Match(PC_XYZ &XPC, PC_XYZ &YPC) = 0;

	virtual void GetResMat(Eigen::MatrixXf &resMat) = 0;

	float GetWight() { return m_W; }

protected:
	//计算点云的缩放参数
	virtual void GetScaleParam(PC_XYZ &XPC, PC_XYZ &YPC);

	//点云去中心化
	virtual void PCCentralization(PC_XYZ &XPC, PC_XYZ &YPC);

	//数据转化
	virtual void PCToMat(PC_XYZ &pc, Eigen::MatrixXf &mat, P_XYZ &gravity);

	//计算P
	virtual void ComputeP();

	//初始化sigma
	virtual void InitSigma(PC_XYZ &XPC, PC_XYZ &YPC);

	//初始化计算
	virtual void InitCompute(PC_XYZ &XPC, PC_XYZ &YPC);

protected:	
	int m_MaxIter;
	float m_Eps;
	float m_W;   //噪声权重
	float m_Sigma_2;
	float m_Np;
	int M, N;

	//归一化参数
	float m_ScaleX = 500;
	float m_ScaleY = 100;
	float m_ScaleZ = 100;

	//重心
	P_XYZ m_GX;
	P_XYZ m_GY;

	Eigen::MatrixXf m_XMat;
	Eigen::MatrixXf m_YMat;
	Eigen::MatrixXf m_PMat;
	Eigen::MatrixXf m_ResMat;
};

