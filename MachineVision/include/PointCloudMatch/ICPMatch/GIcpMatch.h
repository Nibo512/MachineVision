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
	//������Э�������
	void ComputePtCovarMat(P_XYZ *pSrc, vector<int> &PIdx, Eigen::Matrix3f &covarMat);

	//����Э�������
	void CalCovarMats(PC_XYZ &srcPC, vector<Eigen::Matrix3f> &covaMats, int k);

private:
	vector<Eigen::Matrix3f> m_TgtCovarMats;
	vector<Eigen::Matrix3f> m_SrcCovarMats;
};