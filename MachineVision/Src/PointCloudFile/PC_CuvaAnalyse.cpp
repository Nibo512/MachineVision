#include "../../include/PointCloudFile/PC_CuvaAnalyse.h"
#include <Eigen/src/Eigenvalues/SelfAdjointEigenSolver.h>
#include "../../include/PointCloudFile/PointCloudOpr.h"
#include "../../include/BaseOprFile/MathOpr.h"
#include "../../include/PointCloudFile/MLSSmooth.h"
#include "../../include/PointCloudFile/PC_Filter.h"

//拟合多项式=========================================================================
void FitQuadPoly(PC_XYZ& srcPC, Eigen::MatrixXf& res)
{
	Eigen::MatrixXf A = Eigen::MatrixXf::Zero(5, 5);
	Eigen::MatrixXf B = Eigen::MatrixXf::Zero(5, 1);
	int ptNum = srcPC.size();
	for (int i = 0; i < srcPC.size(); ++i)
	{
		float x = srcPC[i].x;
		float y = srcPC[i].y;
		float z = srcPC[i].z;
		float x2 = x * x;
		float xy = x * y;
		float y2 = y * y;

		A(0, 0) += x2 * x2; A(0, 1) += x2 * xy; A(0, 2) += x2 * y2; A(0, 3) += x2 * x; A(0, 4) += x2 * y;
		A(1, 1) += xy * xy; A(1, 2) += xy * y2; A(1, 3) += xy * x; A(1, 4) += xy * y;
		A(2, 2) += y2 * y2; A(2, 3) += y2 * x; A(2, 4) += y * y2;
		A(3, 3) += x * x;     A(3, 4) += x * y;
		A(4, 4) += y * y;
		B(0, 0) += x2 * z; B(1, 0) += xy * z; B(2, 0) += y2 * z; B(3, 0) += x * z; B(4, 0) += y * z;
	}
	A(1, 0) = A(0, 1);
	A(2, 0) = A(0, 2); A(2, 1) = A(1, 2);
	A(3, 0) = A(0, 3); A(3, 1) = A(1, 3); A(2, 3) = A(3, 2);
	A(4, 0) = A(0, 4); A(4, 1) = A(1, 4); A(4, 2) = A(2, 4); A(4, 3) = A(3, 4);

	res = A.inverse() * B;
}
//===================================================================================

//求解极值点=========================================================================
void ComputeExtPt(Eigen::MatrixXf& conf, P_XYZ& extPt)
{
	//cout << conf << endl;
	float inv = 4.0f * conf(0, 0) * conf(2, 0) - conf(1, 0) * conf(1, 0);
	float x_ = conf(1, 0) * conf(4, 0) - 2.0f * conf(2, 0) * conf(3, 0);
	float y_ = conf(1, 0) * conf(3, 0) - 2.0f * conf(0, 0) * conf(4, 0);
	extPt.x = x_ / inv; extPt.y = y_ / inv;
	extPt.z = conf(0, 0) * extPt.x * extPt.x + conf(1, 0) * extPt.x * extPt.y 
		+ conf(2, 0) * extPt.y * extPt.y + conf(3, 0) * extPt.x + conf(4, 0) * extPt.y;
}
//===================================================================================

//曲率分类===========================================================================
SURFTYPE CuvaClass(Eigen::MatrixXf& conf)
{
	//高斯曲率
	float K = 4.0f * conf(0) * conf(2) - conf(1) * conf(1);
	//平均曲率
	float H = conf(0) + conf(2);
	if (abs(K) < CUVATHRES && abs(H) < CUVATHRES)
		return SURFTYPE::PLANE;
	if (abs(K) < CUVATHRES && H > CUVATHRES)
		return SURFTYPE::RIDGE;
	if (abs(K) < CUVATHRES && H < -CUVATHRES)
		return SURFTYPE::VALLEY;
	if (K < -CUVATHRES && H > CUVATHRES)
		return SURFTYPE::SADDLERIDGE;
	if (K < -CUVATHRES && H < -CUVATHRES)
		return SURFTYPE::SADDLEVALLEY;
	if (K > CUVATHRES && H > CUVATHRES)
		return SURFTYPE::PEAK;
	if (K > CUVATHRES && H < -CUVATHRES)
		return SURFTYPE::TRAP;
	if (K < -CUVATHRES && abs(H) < CUVATHRES)
		return SURFTYPE::MINPOINT;
}
//===================================================================================

//曲率分析===========================================================================
void CuvaAnalyse(PC_XYZ& pc, PC_XYZ& dstPC)
{
	int ptNum = pc.size();
	KdTreeFLANN<P_XYZ> kdtree;
	kdtree.setInputCloud(pc.makeShared());

	PC_XYZ planePC, ridgePC;
	for (int i = 0; i < ptNum; ++i)
	{
		vector<int> PIdx;
		vector<float> PDist;
		kdtree.nearestKSearch(pc[i], 25, PIdx, PDist);
		PC_XYZ idxPC;
		PC_ExtractPC(pc, PIdx, idxPC);

		//计算法向量
		Eigen::Vector3f normals;
		P_XYZ gravity;
		CalNormalAndGravity(idxPC, normals, gravity);

		//计算局部变换矩阵-----没有平移原点
		Eigen::Matrix4f locTransMat = Eigen::Matrix4f::Identity();
		CalLocCoordSys(normals, gravity, locTransMat);
		pcl::transformPointCloud(idxPC, idxPC, locTransMat);

		Eigen::MatrixXf res;
		FitQuadPoly(idxPC, res);

		if (CuvaClass(res) == SURFTYPE::PLANE)
		{
			P_XYZ featurePt;
			Eigen::Matrix4f locTransMatInv = locTransMat.inverse();
	/*		pcl::transformPointCloud(idxPC, idxPC, locTransMatInv);*/
			PC_TransLatePoint(locTransMatInv, idxPC[0], featurePt);
			planePC.push_back(featurePt);
		/*	planePC += idxPC;*/
		}
		if (CuvaClass(res) == SURFTYPE::RIDGE)
		{
			P_XYZ featurePt;
			Eigen::Matrix4f locTransMatInv = locTransMat.inverse();
			//pcl::transformPointCloud(idxPC, idxPC, locTransMatInv);
			PC_TransLatePoint(locTransMatInv, idxPC[0], featurePt);
			ridgePC.push_back(featurePt);
			//ridgePC += idxPC;
		}
	}
	pcl::visualization::PCLVisualizer viewer("srcPC");
	viewer.addCoordinateSystem(2);
	pcl::visualization::PointCloudColorHandlerCustom<P_XYZ> red(planePC.makeShared(), 255, 0, 0); //设置点云颜色
	viewer.addPointCloud(planePC.makeShared(), red, "planePC");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "planePC");

	pcl::visualization::PointCloudColorHandlerCustom<P_XYZ> green(ridgePC.makeShared(), 0, 255, 0); //设置点云颜色
	viewer.addPointCloud(ridgePC.makeShared(), green, "ridgePC");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "ridgePC");

	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}
}
//===================================================================================

//曲率分析测试
void CuvaAnalyseTest()
{
	string pc_rotpath = "D:/data/变形测试点云/EV/m_SrcPC_14.ply";
	PC_XYZ srcPC;
	pcl::io::loadPLYFile(pc_rotpath, srcPC);

	PC_XYZ samplePC;
	PC_DownSample(srcPC, samplePC, 1.5, 0);

	PC_XYZ dstPC;
	CuvaAnalyse(samplePC, dstPC);

	pcl::visualization::PCLVisualizer viewer("srcPC");
	viewer.addCoordinateSystem(2);
	//显示轨迹

	pcl::visualization::PointCloudColorHandlerCustom<P_XYZ> red(samplePC.makeShared(), 255, 0, 0); //设置点云颜色
	viewer.addPointCloud(samplePC.makeShared(), red, "samplePC");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "samplePC");

	pcl::visualization::PointCloudColorHandlerCustom<P_XYZ> green(dstPC.makeShared(), 0, 255, 0); //设置点云颜色
	viewer.addPointCloud(dstPC.makeShared(), green, "dstPC");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "dstPC");

	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}
}
