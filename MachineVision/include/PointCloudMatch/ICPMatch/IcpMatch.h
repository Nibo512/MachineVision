#pragma once

#include "../../BaseOprFile/utils.h"

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

class ICP
{
public:
	ICP():m_MaxPPDist(50), m_MaxIter(50), m_Eps(1e-6), m_Loss(0.0f), m_EffPtNum(0)
	{
	};

protected:
	//寻找最邻近点
	void FindKnnPair();

	//设置最大迭代次数
	void SetrMaxIterK(int iter) { m_MaxIter = iter; }

	//设置对应点距离
	void SetPPMaxDist(float maxPPDist) { m_MaxPPDist = maxPPDist; }

	//设置终止条件
	void SetEps(float eps) { m_Eps = eps; }

protected:
	int m_MaxIter;

	float m_MaxPPDist;

	float m_Loss;

	float m_Eps;

	int m_EffPtNum;

	vector<PairIdx> m_PairIdxes;

	KdTreeFLANN<P_XYZ> m_TgtKdTree;

	PC_XYZ m_SrcPC;
	PC_XYZ m_TgtPC;
};