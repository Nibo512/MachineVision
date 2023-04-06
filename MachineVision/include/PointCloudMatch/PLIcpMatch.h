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
	//ƥ�����
	float JC_PCMatch(PC_XYZ &srcPC, PC_XYZ &tgtPC, Eigen::Matrix4f &transMat);

	//��������������
	void SetrMaxIterK(int iter) { m_MaxIter = iter; }

	//���ö�Ӧ�����
	void SetPPMaxDist(float maxPPDist) { m_MaxPPDist = maxPPDist; }

	//������ֹ����
	void SetEps(float eps) { m_Eps = eps; }

private:
	//Ѱ�����ڽ���
	void JC_FindKnnPair();

	//����任����---����任
	void JC_ComputeAffineTransMat(Eigen::Matrix4f &transMat);

	//���Ա任
	void JC_ComputeRigidTranMat(Eigen::Matrix4f &transMat);

	//����������
	void JC_ComputeTransMatWithNormal(Eigen::Matrix4f &transMat);

	//����loss
	bool ComputeLoss();

	//����ƥ�����
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
