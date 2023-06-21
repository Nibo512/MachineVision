#include "../../../include/PointCloudMatch/CPDMatch/NonRigidCPD.h"
#include <boost/algorithm/string/split.hpp>
#include <Eigen/Eigenvalues>
#include "../../../include/PointCloudFile/PC_Filter.h"

//初始化非刚性变换计算==========================================================
void NonRigidCPD::InitNonRigidCompute(PC_XYZ &XPC, PC_XYZ &YPC)
{
	InitCompute(XPC, YPC);
	m_Beta = 16.0f; m_Lamda = 8.0f;
	m_GMat = Eigen::MatrixXf::Identity(M, M);
	m_WMat = Eigen::MatrixXf::Zero(M, 3);
	m_ResMat = Eigen::MatrixXf::Zero(M, 3);
	ConstructGMat();
}
//==============================================================================

//析构==========================================================================
NonRigidCPD::~NonRigidCPD()
{
}
//==============================================================================

//构造G矩阵=====================================================================
void NonRigidCPD::ConstructGMat()
{
	float beta2 = -1.0f / (2.0f * m_Beta);
	for (int m1 = 0; m1 < M; ++m1)
	{
		for (int m2 = 0; m2 < m1; ++m2)
		{
			float diff_x = m_YMat(m1, 0) - m_YMat(m2, 0);
			float diff_y = m_YMat(m1, 1) - m_YMat(m2, 1);
			float diff_z = m_YMat(m1, 2) - m_YMat(m2, 2);
			m_GMat(m1, m2) = std::exp((diff_x * diff_x + diff_y * diff_y + diff_z * diff_z) * beta2);
			m_GMat(m2, m1) = m_GMat(m1, m2);
		}
	}

	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> es(m_GMat);
	Eigen::MatrixXf eigenValue = es.eigenvalues();
	Eigen::MatrixXf eigenVector = es.eigenvectors();

	m_LowRankN = std::pow(float(M), 0.4);
	m_GDig = Eigen::MatrixXf::Zero(m_LowRankN, m_LowRankN);
	m_GQ = Eigen::MatrixXf::Zero(M, m_LowRankN);

	for (int m = 0; m < m_LowRankN; ++m)
	{
		m_GDig(m, m) = 1.0f / eigenValue(M - m - 1, 0);
		m_GQ.col(m) = eigenVector.col(M - m - 1);
	}
}
//==============================================================================

//计算A=========================================================================
void NonRigidCPD::ComputeA(Eigen::MatrixXf &A, Eigen::MatrixXf &dPM, float c)
{
	c = 1.0f / c;
	Eigen::MatrixXf P1Q = Eigen::MatrixXf::Zero(M, m_LowRankN);
	for (int m = 0; m < M; ++m)
	{
		P1Q.row(m) = dPM(m) * m_GQ.row(m);
	}
	Eigen::MatrixXf GQ_T = m_GQ.transpose();
	A = m_GDig + c * GQ_T * P1Q;
	A = c * c * P1Q * A.inverse() * GQ_T;
	A = -A;
	for (int m = 0; m < M; ++m)
	{
		A(m, m) += c;
	}
}
//==============================================================================

//计算非刚性变换矩阵============================================================
void NonRigidCPD::ComputeNonRigidTranMat()
{
	m_Np = m_PMat.sum();
	//求解W矩阵
	Eigen::MatrixXf dPM = (m_PMat.rowwise().sum());
	Eigen::MatrixXf A(M, M);
	ComputeA(A, dPM, m_Lamda * m_Sigma_2);

	Eigen::MatrixXf PMY(M, 3);
//#pragma omp parallel for
	for (int m = 0; m < M; ++m)
	{
		PMY.row(m) = dPM(m) * m_YMat.row(m);
	}
	Eigen::MatrixXf B = m_PMat * m_XMat - PMY;
	m_WMat = A * B;

	NonRigidTransPC();
//#pragma omp parallel for
	for (int m = 0; m < M; ++m)
	{
		PMY.row(m) = dPM(m) * m_YMat.row(m);
	}
	//计算sigma
	Eigen::MatrixXf dPN = (m_PMat.colwise().sum());

	Eigen::MatrixXf dPNX = Eigen::MatrixXf(N, 3);
	for (int n = 0; n < N; ++n)
	{
		dPNX.row(n) = dPN(n) * m_XMat.row(n);
	}
	Eigen::MatrixXf sigma1 = m_XMat.transpose() * dPNX;
	Eigen::MatrixXf sigma2 = (m_PMat * m_XMat).transpose() * m_YMat;
	Eigen::MatrixXf sigma3 = m_YMat.transpose() * PMY;
	m_Sigma_2 = (sigma1.trace() - 2.0f * sigma2.trace() + sigma3.trace()) / (3.0f * m_Np);
	if (m_Sigma_2 <= 0)
		m_Sigma_2 = 1e-8;
}
//==============================================================================

//点云非刚性变换================================================================
void NonRigidCPD::NonRigidTransPC()
{
	m_YMat += m_GMat * m_WMat;
}
//==============================================================================

//非刚性配准====================================================================
void NonRigidCPD::Match(PC_XYZ &XPC, PC_XYZ &YPC)
{
	InitNonRigidCompute(XPC, YPC);
	int iter = 0;

	double t1 = cv::getTickCount();
	while (m_Sigma_2 > 1e-8 && iter < m_MaxIter)
	{
		//E-Step
		ComputeP();
		ComputeNonRigidTranMat();

		m_ResMat += m_WMat;
		iter++;
	}
	double t2 = cv::getTickCount();
	cout << (t2 - t1) / cv::getTickFrequency() << endl;
}
//==============================================================================

//获取匹配结果==================================================================
void NonRigidCPD::GetResMat(Eigen::MatrixXf &resMat)
{
	resMat = m_GMat * m_ResMat;
	for (int m = 0; m < M; ++m)
	{
		resMat(m, 0) = resMat(m, 0) * m_ScaleX + m_GX.x - m_GY.x;
		resMat(m, 1) = resMat(m, 1) * m_ScaleY + m_GX.y - m_GY.y;
		resMat(m, 2) = resMat(m, 2) * m_ScaleZ + m_GX.z - m_GY.z;
	}
}
//==============================================================================

void TestNonRigidMatch()
{
	PC_XYZ XPC_, YPC_;
	pcl::io::loadPLYFile("D:/data/变形测试点云/EV/3.ply", YPC_);
	pcl::io::loadPLYFile("D:/data/变形测试点云/EV/2.ply", XPC_);

	PC_XYZ XPC, YPC;
	PC_VoxelGrid(XPC_, XPC, 5);
	PC_VoxelGrid(YPC_, YPC, 5);

	//string txtFileX = "C:/Users/Administrator/Downloads/CPD—Data/data2/fish.csv";
	//string txtFileY = "C:/Users/Administrator/Downloads/CPD—Data/data2/fish_distorted.csv";
	//TxtToPC(txtFileX, XPC, 0);
	//TxtToPC(txtFileY, YPC, 0);

	float w = 0.5;
	NonRigidCPD cpdReg(0.5f, 30, 1e-8);
	cpdReg.Match(XPC, YPC);
	Eigen::MatrixXf resMat;
	cpdReg.GetResMat(resMat);

	PC_XYZ outPt;
	outPt.resize(YPC.size());
	P_XYZ *pPCData = outPt.points.data();
	for (int i = 0; i < YPC.size(); ++i)
	{
		pPCData[i].x = YPC[i].x + resMat(i, 0);
		pPCData[i].y = YPC[i].y + resMat(i, 1);
		pPCData[i].z = YPC[i].z + resMat(i, 2);
	}

	pcl::visualization::PCLVisualizer viewer;
	viewer.addCoordinateSystem(10);
	//显示轨迹
	pcl::visualization::PointCloudColorHandlerCustom<P_XYZ> red(XPC.makeShared(), 255, 0, 0); //设置点云颜色
	viewer.addPointCloud(XPC.makeShared(), red, "XPC");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "XPC");

	pcl::visualization::PointCloudColorHandlerCustom<P_XYZ> green(outPt.makeShared(), 0, 255, 0); //设置点云颜色
	viewer.addPointCloud(outPt.makeShared(), green, "outPt");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "outPt");

	pcl::visualization::PointCloudColorHandlerCustom<P_XYZ> blue(YPC.makeShared(), 0, 0, 255); //设置点云颜色
	viewer.addPointCloud(YPC.makeShared(), blue, "YPC");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "YPC");

	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}
}
