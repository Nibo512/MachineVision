#pragma once
#include "IcpMatch.h"

class GICP : public ICP
{
public:
	GICP() :ICP()
	{
		m_TgtCovarMats.resize(0);
		m_SrcCovarMats.resize(0);
	};

private:
	//计算点的协方差矩阵
	void ComputePtCovarMat(P_XYZ *pSrc, vector<int> &PIdx, Eigen::Matrix3f &covarMat);

	//计算协方差矩阵
	void CalCovarMats(PC_XYZ &srcPC, vector<Eigen::Matrix3f> &covaMats, int k);

private:
	vector<Eigen::Matrix3f> m_TgtCovarMats;
	vector<Eigen::Matrix3f> m_SrcCovarMats;
};