#include "../../include/PointCloudFile/PointCloudOpr.h"
#include "../../include/BaseOprFile/MathOpr.h"
#include "../../include/FitShapeFile/ComputePts.h"
#include <Eigen/src/Eigenvalues/SelfAdjointEigenSolver.h>

//������Ƶ���С��Χ��==============================================================
void PC_ComputeOBB(const PC_XYZ &srcPC, PC_XYZ &obb)
{
	Eigen::Vector4f pcaCentroid;
	pcl::compute3DCentroid(srcPC, pcaCentroid);
	Eigen::Matrix3f covariance;
	pcl::computeCovarianceMatrixNormalized(srcPC, pcaCentroid, covariance);
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
	Eigen::Matrix3f eigenVectorsPCA = eigen_solver.eigenvectors();

	Eigen::Matrix4f transMat = Eigen::Matrix4f::Identity();
	transMat.block<3, 3>(0, 0) = eigenVectorsPCA.transpose();;
	transMat.block<3, 1>(0, 3) = -1.0f * transMat.block<3, 3>(0, 0) * (pcaCentroid.head<3>());

	PC_XYZ::Ptr transPC(new PC_XYZ);
	pcl::transformPointCloud(srcPC, *transPC, transMat);

	P_XYZ min_p, max_p;
	pcl::getMinMax3D(*transPC, min_p, max_p);
	PC_XYZ::Ptr tran_box(new PC_XYZ);
	tran_box->points.resize(8);
	tran_box->points[0] = min_p;
	tran_box->points[1] = { max_p.x, min_p.y,  min_p.z };
	tran_box->points[2] = { max_p.x, max_p.y,  min_p.z };
	tran_box->points[3] = { min_p.x, max_p.y,  min_p.z };

	tran_box->points[4] = { min_p.x, min_p.y,  max_p.z };
	tran_box->points[5] = { max_p.x, min_p.y,  max_p.z };
	tran_box->points[6] = max_p;
	tran_box->points[7] = { min_p.x, max_p.y,  max_p.z };
	pcl::transformPointCloud(*tran_box, obb, transMat.inverse());
}
//==================================================================================

//��ȡ����==========================================================================
void PC_ExtractPC(const PC_XYZ &srcPC, vector<int> &indexes, PC_XYZ &dstPC)
{
	uint size_src = srcPC.size();
	uint size_idx = indexes.size();
	if (size_idx == 0 || size_src == 0 || size_src < size_idx)
		return;
	dstPC.resize(size_idx);
	for (uint i = 0; i < size_idx; ++i)
	{
		dstPC[i] = srcPC[indexes[i]];
	}
}
//===================================================================================

//����ֱ��ͶӰƽ��===================================================================
void PC_LineProjSmooth(const PC_XYZ &srcPC, PC_XYZ &dstPC, int size, double thresVal)
{
	size_t length = srcPC.size();
	KdTreeFLANN<P_XYZ> kdtree;
	kdtree.setInputCloud(srcPC.makeShared());
	vector<cv::Point3d> cvp_(size);
	dstPC.resize(length);
	for (int i = 0; i < length; ++i)
	{
		vector<int> PIdx(0);
		vector<float> PDist(0);
		const P_XYZ &p_ = srcPC[i];
		kdtree.nearestKSearch(p_, size, PIdx, PDist);
		for (int j = 0; j < size; ++j)
		{
			cvp_[j].x = srcPC[PIdx[j]].x;
			cvp_[j].y = srcPC[PIdx[j]].y;
			cvp_[j].z = srcPC[PIdx[j]].z;
		}
		cv::Vec6f line;
		cv::fitLine(cvp_, line, cv::DIST_L2, 0, 0.01, 0.01);
		P_XYZ projPt;
		PC_PtProjLinePt(p_, line, projPt);
		float dist = std::powf(projPt.x - p_.x, 2) + std::powf(projPt.y - p_.y, 2) + std::powf(projPt.z - p_.z, 2);
		dstPC[i] = dist > thresVal ? projPt : p_;
	}
}
//===================================================================================

//������Ƶ�����=====================================================================
void PC_GetPCGravity(PC_XYZ &srcPC, P_XYZ &gravity)
{
	int point_num = srcPC.size();
	if (point_num == 0)
		return;
	float sum_x = 0.0, sum_y = 0.0, sum_z = 0.0;
	for (int i = 0; i < point_num; ++i)
	{
		sum_x += srcPC[i].x;
		sum_y += srcPC[i].y;
		sum_z += srcPC[i].z;
	}
	gravity.x = sum_x / point_num;
	gravity.y = sum_y / point_num;
	gravity.z = sum_z / point_num;
}
//===================================================================================

//������ͶӰ��XY��άƽ��=============================================================
void PC_ProjectToXY(PC_XYZ &srcPC, cv::Mat &xyPlane)
{
	P_XYZ min_pt, max_pt;
	int scale = 1;
	pcl::getMinMax3D(srcPC, min_pt, max_pt);
	int imgW = (max_pt.x - min_pt.x) * scale + 10;
	int imgH = (max_pt.y - min_pt.y) * scale + 10;
	int z_Scalar = 255 / (max_pt.z - min_pt.z);

	xyPlane = cv::Mat(cv::Size(imgW, imgH), CV_8UC1, cv::Scalar(0));
	uchar* pImagXY = xyPlane.ptr<uchar>();
	for (int i = 0; i < srcPC.size(); ++i)
	{
		P_XYZ& p_ = srcPC[i];
		int index_x = (p_.x - min_pt.x) * scale + 5;
		int index_y = (p_.y - min_pt.y) * scale + 5;
		pImagXY[index_y * imgW + index_x] = (p_.z - min_pt.z) * z_Scalar;
	}
}
//===================================================================================

//������Ƶķ�����===================================================================
void PC_ComputePCNormal(PC_XYZ &srcPC, PC_N &normals, float radius)
{
	if (srcPC.empty())
		return;
	pcl::NormalEstimation<P_XYZ, pcl::Normal> normal_est;
	normal_est.setInputCloud(srcPC.makeShared());
	normal_est.setRadiusSearch(radius);
	pcl::search::KdTree<P_XYZ>::Ptr kdtree(new pcl::search::KdTree<P_XYZ>);
	normal_est.setSearchMethod(kdtree);
	normal_est.compute(normals);
}
//===================================================================================

//������Ƶ�Э�������===============================================================
void PC_ComputeCovMat(PC_XYZ &pc, Mat &covMat, P_XYZ &gravity)
{
	if (pc.empty())
		return;
	if (covMat.size() != cv::Size(3, 3))
		covMat = Mat(cv::Size(3, 3), CV_32FC1, cv::Scalar(0));
	int point_num = pc.size();

	PC_GetPCGravity(pc, gravity);

	vector<float> ori_x(point_num), ori_y(point_num), ori_z(point_num);
	for (int i = 0; i < point_num; ++i)
	{
		P_XYZ& p_ = pc[i];
		ori_x[i] = p_.x - gravity.x;
		ori_y[i] = p_.y - gravity.y;
		ori_z[i] = p_.z - gravity.z;
	}

	float *pCovMat = covMat.ptr<float>(0);
	for (int i = 0; i < point_num; ++i)
	{
		pCovMat[0] += (ori_x[i] * ori_x[i]);
		pCovMat[4] += (ori_y[i] * ori_y[i]);
		pCovMat[8] += (ori_z[i] * ori_z[i]);

		pCovMat[1] += (ori_x[i] * ori_y[i]);
		pCovMat[5] += (ori_y[i] * ori_z[i]);
		pCovMat[2] += (ori_z[i] * ori_x[i]);
	}
	pCovMat[3] = pCovMat[1];
	pCovMat[6] = pCovMat[2];
	pCovMat[7] = pCovMat[5];
}
//===================================================================================
