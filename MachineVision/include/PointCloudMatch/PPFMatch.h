#pragma once
#include "../BaseOprFile/utils.h"
#include <hash_map>
#include <unordered_map>

const float eps_1 = -0.99999999;
const float eps1 = 0.99999999;

typedef struct PPFCELL
{
	uint ref_i;
	float ref_alpha;
	PPFCELL() :ref_i(0), ref_alpha(0.0f)
	{}
}PPFCELL;

typedef struct PPFFEATRUE
{
	int dist;
	int ang_N1D;
	int ang_N2D;
	int ang_N1N2;
	PPFFEATRUE() :dist(0), ang_N1D(0),
		ang_N2D(0), ang_N1N2(0)
	{}
}PPFFEATRUE;

//typedef struct PPFMODEL
//{
//	vector<cv::Mat> refTransMat;
//	uint numAng;
//	float alphStep;
//	float distStep;
//	float angThres;
//	float distThres;
//	hash_map<string, vector<PPFCELL>> hashMap;
//	PPFMODEL() :numAng(5.0f), distStep(0.1f),
//		angThres(0.0f), distThres(0.0f)
//	{
//		alphStep = (float)CV_2PI / numAng;
//		angThres = (float)CV_2PI / alphStep;
//		refTransMat.resize(0);
//	}
//}PPFMODEL;

typedef struct PPFPose
{
	Eigen::Matrix4f transMat;
	uint votes;
	uint ref_i;
	uint i_;
	PPFPose() :votes(0), ref_i(0), i_(0)
	{}
}PPFPose;

//struct HashKey : public std::pair <int, std::pair <int, std::pair <int, int> > >
//{
//	HashKey(int a, int b, int c, int d)
//	{
//		this->first = a;
//		this->second.first = b;
//		this->second.second.first = c;
//		this->second.second.second = d;
//	}
//	int operator()(const HashKey& s) const noexcept
//	{
//		int h1 = std::hash<int>{} (s.first);
//		int h2 = std::hash<int>{} (s.second.first);
//		int h3 = std::hash<int>{} (s.second.second.first);
//		int h4 = std::hash<int>{} (s.second.second.second);
//		return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3);
//	}
//};

class PPFMATCH
{
public:
	PPFMATCH(float angleStep, float distStep) :m_AngleStep(angleStep), m_DistStep(distStep)
	{
	}

	void CreatePPFModel(PC_XYZ &modelPC, PC_N &model_n);

	void MatchPose(PC_XYZ &testPC, PC_N &testPCN, Eigen::Matrix4f &resTransMat);

private:
	//计算PPF特征
	void ComputePPFFEATRUE(P_XYZ &p1, P_XYZ &p2, P_N &pn1, P_N &pn2, PPFFEATRUE &ppfFEATRUE);

	//构建哈希表
	void CreateHashMap(PPFFEATRUE &ppfFEATRUE, int i, int j, float alpha);

	//罗格里德斯公式
	void RodriguesFormula(P_N &rotAxis, float rotAng, Eigen::Matrix4f &rotMat);

	//计算局部转换矩阵
	void ComputeLocTransMat(P_XYZ &ref_p, P_N &ref_pn, Eigen::Matrix4f &transMat);

	//计算局部坐标系下的alpha
	float ComputeLocalAlpha(P_XYZ &ref_p, P_N &ref_pn, P_XYZ &p_, Eigen::Matrix4f &transMat);

	//重置投票器
	void ResetVoteScheme(vector<vector<int>> &VoteScheme);

	//计算变换矩阵
	void ComputeTransMat(Eigen::Matrix4f &SToGMat, float alpha, const Eigen::Matrix4f &RToGMat, Eigen::Matrix4f &transMat);

private:
	unordered_map<string, vector<PPFCELL>> m_ModelFeatrue;

	float m_AngleStep;
	float m_DistStep;

	PC_XYZ m_ModelPC;
	PC_N m_ModelNormal;
};


//测试程序
void PPFTestProgram();