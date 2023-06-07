#include "../../../include/PointCloudMatch/ICPMatch/GIcpMatch.h"

//计算点的协方差矩阵==============================================================
void GICP::ComputePtCovarMat(P_XYZ* pSrc, vector<int>& PIdx, Eigen::Matrix3f& covarMat)
{
	float sum_x = 0.0f, sum_y = 0.0f, sum_z = 0.0f;
	int ptNum = PIdx.size();
	for (int i = 0; i < ptNum; ++i)
	{
		P_XYZ& pt = pSrc[PIdx[i]];
		sum_x += pt.x; sum_y += pt.y; sum_z += pt.z;
	}
	float invK = 1.0f / float(ptNum);
	sum_x *= invK; sum_y *= invK; sum_z *= invK;
	covarMat = Eigen::Matrix3f::Zero();
	for (int i = 0; i < ptNum; ++i)
	{
		P_XYZ& pt = pSrc[PIdx[i]];
		float x_ = pt.x - sum_x;
		float y_ = pt.y - sum_y;
		float z_ = pt.z - sum_z;
		covarMat(0, 0) += x_ * x_; covarMat(0, 1) += x_ * y_; covarMat(0, 2) += x_ * z_;
		covarMat(1, 1) += y_ * y_; covarMat(1, 2) += y_ * z_;
		covarMat(2, 2) += z_ * z_;
	}
	covarMat(1, 0) = covarMat(0, 1); covarMat(2, 0) = covarMat(0, 2); covarMat(2, 1) = covarMat(1, 2);
}
//================================================================================

//计算协方差矩阵==================================================================
void GICP::CalCovarMats(PC_XYZ& srcPC, vector<Eigen::Matrix3f>& covarMats, int k)
{
	int ptNum = srcPC.size();
	if (k > ptNum)
		return;
	KdTreeFLANN<P_XYZ> kdtree;
	kdtree.setInputCloud(srcPC.makeShared());
	covarMats.resize(ptNum);
	P_XYZ* pSrc = srcPC.points.data();
	for (int i = 0; i < ptNum; ++i)
	{
		vector<int> PIdx;
		vector<float> PDist;
		kdtree.nearestKSearch(pSrc[i], k, PIdx, PDist);
		ComputePtCovarMat(pSrc, PIdx, covarMats[i]);
	}
}
//================================================================================