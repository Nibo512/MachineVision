#pragma once
#include "CPDMatch.h"

class RigidCPD :public CPD
{
public:
	RigidCPD(float w = 0.2f, int maxIter = 50, float eps = 1e-6f)
		:CPD(w, maxIter, eps)
	{}

	//配准
	void Match(PC_XYZ &XPC, PC_XYZ &YPC);

	void GetResMat(Eigen::MatrixXf &resMat);

	float GetMagnification()
	{
		return m_S;
	}

private:
	//初始化刚性变换计算
	void InitRigidCompute(PC_XYZ &XPC, PC_XYZ &YPC);

	//计算XTip, YTip
	void ComputeXTipYTip();

	//计算刚性变换矩阵
	void ComputeRigidTranMat();

	//计算仿射变换矩阵
	void ComputeAffineTranMat();

	//点云刚性变换
	void RigidTransPC();

	//刚性变换中使用
	float m_S;   //放大率
	Eigen::Matrix3f m_RotMat;  //旋转矩阵
	Eigen::MatrixXf m_TranMat; //平移矩阵
	Eigen::MatrixXf m_XTipMat;
	Eigen::MatrixXf m_YTipMat;
	Eigen::MatrixXf m_Ux;
	Eigen::MatrixXf m_Uy;
};

void TestRigidMatch();
