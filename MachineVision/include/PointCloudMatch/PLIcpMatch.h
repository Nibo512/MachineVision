#pragma once
#include "../BaseOprFile/utils.h"

struct PairIdx
{
	int srcIdx;
	int tgtIdx;
	PairIdx()
	{}
	PairIdx(int srcIdx_, int tgtIdx_)
	{
		srcIdx = srcIdx_;
		tgtIdx = tgtIdx_;
	}
};


class JCMATCH
{
public:
	JCMATCH():m_MaxPPDist(50), m_MaxIter(50), m_Eps(1e-6), m_Loss(0.0f), m_EffPtNum(0)
	{
	};

public:
	//匹配程序
	float JC_PCMatch(PC_XYZ &srcPC, PC_XYZ &tgtPC, Eigen::Matrix4f &transMat);

	//设置最大迭代次数
	void SetrMaxIterK(int iter) { m_MaxIter = iter; }

	//设置对应点距离
	void SetPPMaxDist(float maxPPDist) { m_MaxPPDist = maxPPDist; }

	//设置终止条件
	void SetEps(float eps) { m_Eps = eps; }

private:
	//寻找最邻近点
	void JC_FindKnnPair();

	//计算变换矩阵---仿射变换
	void JC_ComputeAffineTransMat(Eigen::Matrix4f &transMat);

	//刚性变换
	void JC_ComputeRigidTranMat(Eigen::Matrix4f &transMat);

	//带法向量的
	void JC_ComputeTransMatWithNormal(Eigen::Matrix4f &transMat);

	//计算loss
	bool ComputeLoss();

	//计算匹配分数
	float ComputeFitScore();

private:
	KdTreeFLANN<P_XYZ> m_TgtKdTree;
	int m_MaxIter;

	PC_XYZ m_SrcPC;
	PC_XYZ m_TgtPC;

	vector<PairIdx> m_PairIdxes;

	PC_N m_TgtPtVecs;

	float m_MaxPPDist;

	float m_Loss;

	float m_Eps;

	int m_EffPtNum;
};

void PLIcpMatchTest();
