#include "../../include/PointCloudFile/MLSSmooth.h"
#include <Eigen/src/Eigenvalues/SelfAdjointEigenSolver.h>
#include "../../include/PointCloudFile/PointCloudOpr.h"
#include "../../include/BaseOprFile/MathOpr.h"

double t = 0.0;

//计算法向量以及重心=================================================================
void CalNormalAndGravity(PC_XYZ& srcPC, Eigen::Vector3f& normals, P_XYZ& gravity)
{
	if (srcPC.size() < 4)
		return;
	P_XYZ* pPts = srcPC.points.data();
	int ptNum = srcPC.size();
	gravity.x = 0.0f, gravity.y = 0.0f, gravity.z = 0.0f;
	for (int i = 0; i < ptNum; ++i)
	{
		gravity.x += pPts[i].x;
		gravity.y += pPts[i].y;
		gravity.z += pPts[i].z;
	}
	float w_sum = 1.0f / std::max((float)ptNum, 1e-8f);
	gravity.x *= w_sum; gravity.y *= w_sum; gravity.z *= w_sum;

	Eigen::Matrix3f A = Eigen::Matrix3f::Zero();
	for (int i = 0; i < ptNum; ++i)
	{
		float x_ = pPts[i].x - gravity.x;
		float y_ = pPts[i].y - gravity.y;
		float z_ = pPts[i].z - gravity.z;
		A(0) += x_ * x_;
		A(4) += y_ * y_;
		A(8) += z_ * z_;
		A(1) += x_ * y_;
		A(2) += x_ * z_;
		A(5) += y_ * z_;
	}
	A(3) = A(1); A(6) = A(2); A(7) = A(5);
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(A);
	Eigen::Matrix3f vect = es.eigenvectors();
	normals(0) = vect(0, 0); normals(1) = vect(1, 0); normals(2) = vect(2, 0);
}
//===================================================================================

//计算局部转换矩阵===================================================================
void CalLocCoordSys(Eigen::Vector3f& normals, P_XYZ& gravity, Eigen::Matrix4f& locTransMat)
{
	float cosBeta = std::sqrt(normals(1) * normals(1) + normals(2) * normals(2));
	float cosAlpha = normals(2) / cosBeta;
	float sinAlpha = normals(1) / cosBeta;
	float sinBeta = normals(0);

	Eigen::Matrix4f Ry = Eigen::Matrix4f::Identity();
	Ry(0, 0) = cosBeta; Ry(0, 2) = -sinBeta; Ry(2, 0) = sinBeta; Ry(2, 2) = cosBeta;

	Eigen::Matrix4f Rx = Eigen::Matrix4f::Identity();
	Rx(1, 1) = cosAlpha; Rx(1, 2) = -sinAlpha; Rx(2, 1) = sinAlpha; Rx(2, 2) = cosAlpha;
	locTransMat(0, 3) = -gravity.x; locTransMat(1, 3) = -gravity.y; locTransMat(2, 3) = -gravity.z;
	locTransMat = Ry * Rx * locTransMat;
}
//===================================================================================

//计算权重===========================================================================
void CalPtWeigth(PC_XYZ& pts, vector<float>& w)
{
	int ptNum = pts.size();
	w.resize(ptNum);
	for (int i = 0; i < ptNum; ++i)
	{
		P_XYZ& pt = pts[i];
		w[i] = std::exp(-(pt.x * pt.x + pt.y * pt.y));
	}
}
//===================================================================================

//计算多项式=========================================================================
void CalPolyVec(P_XYZ& pt, vector<float>& polyVec, int order)
{
	int idx_vd = 0;
	int order_i = order + 1;
	float x = pt.x;
	float y = pt.y;
	for (int i = 0; i < order_i; ++i)
	{
		float x_r = i == 0 ? 1 : static_cast<float>(std::pow(x, i));
		int order_k = order - i + 1;
		for (int k = 0; k < order_k; ++k)
		{
			if (i != 0 || k != 0)
			{
				float y_r = k == 0 ? 1 : static_cast<float>(std::pow(y, k));
				polyVec[idx_vd] = x_r * y_r;
				++idx_vd;
			}
		}
	}
}
//===================================================================================

//计算矩阵a、b=======================================================================
void CalMatAB(vector<float>& polyVec, P_XYZ& pt, Eigen::MatrixXf& a_mat,
	Eigen::MatrixXf& b_mat, int order, float w_)
{
	int idx_vd = 0;
	int size = polyVec.size();
	for (int r = 0; r < size; ++r)
	{
		float coff_ = w_ * polyVec[r];
		for (int c = 0; c < r + 1; ++c)
		{
			a_mat(r, c) += coff_ * polyVec[c];
		}
		b_mat(r) += coff_ * pt.z;
	}
}
//===================================================================================

//计算系数矩阵=======================================================================
void CalCoffMat(PC_XYZ& srcPC, Eigen::MatrixXf& coffMat, vector<float> &ws, int order)
{
	int k_order = (order + 1) * (order + 2) / 2 - 1;
	Eigen::MatrixXf a_mat = Eigen::MatrixXf::Zero(k_order, k_order);
	Eigen::MatrixXf b_mat = Eigen::MatrixXf::Zero(k_order, 1);
	int matIdx = 0;
	vector<float> polyVec(k_order);
	for (int pI = 0; pI < srcPC.size(); ++pI)
	{
		P_XYZ& pt = srcPC[pI];
		float w_ = ws[pI];

		//计算多项式
		CalPolyVec(pt, polyVec, order);
		//计算矩阵a，b
		CalMatAB(polyVec, pt, a_mat, b_mat, order, w_);
	}
	for (int r = 0; r < k_order; ++r)
	{
		for (int c = r + 1; c < k_order; ++c)
		{
			a_mat(r, c) = a_mat(c, r);
		}
	}
	coffMat = a_mat.inverse() * b_mat;
}
//===================================================================================

//投影===============================================================================
void PtProjToPoly(P_XYZ& pt, Eigen::MatrixXf& coffMat, P_XYZ& projPt, int order)
{
	int idx_vd = 0;
	int order_i = order + 1;
	float z = 0.0f;
	for (int i = 0; i < order_i; ++i)
	{
		float x_r = i == 0 ? 1 : std::pow(pt.x, i);
		int order_k = order - i + 1;
		for (int k = 0; k < order_k; ++k)
		{
			if (i == 0 && k == 0)
				continue;
			float y_r = k == 0 ? 1 : std::pow(pt.y, k);
			z += coffMat(idx_vd) * x_r * y_r;
			++idx_vd;
		}
	}
	projPt = {pt.x, pt.y, z};
}
//===================================================================================

//MLS平滑============================================================================
void MLSSmooth(PC_XYZ& srcPC, PC_XYZ& dstPC, float radius, int order)
{
	KdTreeFLANN<P_XYZ> kdtree;
	kdtree.setInputCloud(srcPC.makeShared());
	int ptNum = srcPC.size();
	dstPC.resize(ptNum);

	double t1 = cv::getTickCount();
#pragma omp parallel for
	for (int i = 0; i < ptNum; ++i)
	{
		vector<int> PIdx;
		vector<float> PDist;
		kdtree.radiusSearch(srcPC[i], radius, PIdx, PDist);

		PC_XYZ locPC;
		PC_ExtractPC(srcPC, PIdx, locPC);

		//计算法向量
		Eigen::Vector3f normals;
		P_XYZ gravity;
		CalNormalAndGravity(locPC, normals, gravity);

		//计算局部变换矩阵
		Eigen::Matrix4f locTransMat = Eigen::Matrix4f::Identity();
		CalLocCoordSys(normals, gravity, locTransMat);
		pcl::transformPointCloud(locPC, locPC, locTransMat);

		//计算系数矩阵
		vector<float> ws;
		CalPtWeigth(locPC, ws);
		Eigen::MatrixXf coffMat;
		CalCoffMat(locPC, coffMat, ws, order);

		//点投影
		P_XYZ projPt;
		PtProjToPoly(locPC[0], coffMat, projPt, order);
		Eigen::Matrix4f locTransMatInv = locTransMat.inverse();
		PC_TransLatePoint(locTransMatInv, projPt, dstPC[i]);
	}
	double t2 = cv::getTickCount();
	t += t2 - t1;
	cout << t / cv::getTickFrequency() << endl;
}
//===================================================================================

//测试MLS
void TestMLS()
{
	string pc_rotpath = "D:/data/变形测试点云/EV/m_SrcPC_14.ply";
	PC_XYZ srcPC;
	pcl::io::loadPLYFile(pc_rotpath, srcPC);

	PC_XYZ dstPC;
	MLSSmooth(srcPC, dstPC, 2.5, 2);

	pcl::visualization::PCLVisualizer viewer("srcPC");
	viewer.addCoordinateSystem(10);
	//显示轨迹
	pcl::visualization::PointCloudColorHandlerCustom<P_XYZ> red(srcPC.makeShared(), 255, 0, 0); //设置点云颜色
	viewer.addPointCloud(srcPC.makeShared(), red, "srcPC");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "srcPC");

	pcl::visualization::PointCloudColorHandlerCustom<P_XYZ> green(dstPC.makeShared(), 0, 255, 0); //设置点云颜色
	viewer.addPointCloud(dstPC.makeShared(), green, "dstPC");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "dstPC");

	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}
}

