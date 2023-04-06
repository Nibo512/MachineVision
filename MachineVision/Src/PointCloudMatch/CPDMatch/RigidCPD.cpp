#include "../../../include/PointCloudMatch/CPDMatch/RigidCPD.h"
#include <Eigen/Eigenvalues>
#include <boost/algorithm/string/split.hpp>
#include "../../../include/PointCloudFile/PointCloudOpr.h"

//初始化计算====================================================================
void RigidCPD::InitRigidCompute(PC_XYZ &XPC, PC_XYZ &YPC)
{
	InitCompute(XPC, YPC);
	m_RotMat = Eigen::Matrix3f::Identity();
	m_TranMat = Eigen::MatrixXf::Zero(1, 3);
	m_S = 1.0f;
	m_XTipMat = Eigen::MatrixXf::Zero(N, 3);
	m_YTipMat = Eigen::MatrixXf::Zero(M, 3);
}
//==============================================================================

//计算XTip, YTip================================================================
void RigidCPD::ComputeXTipYTip()
{
	//计算Np, ux, uy
	m_Np = m_PMat.sum() + 1e-12f;
	m_Ux = (m_PMat * m_XMat).colwise().sum();
	m_Uy = (m_PMat.transpose() * m_YMat).colwise().sum();
	//计算XTip, YTip
	m_Ux /= m_Np; m_Uy /= m_Np;
	for (int i = 0; i < N; ++i)
	{
		m_XTipMat(i, 0) = m_XMat(i, 0) - m_Ux(0, 0);
		m_XTipMat(i, 1) = m_XMat(i, 1) - m_Ux(0, 1);
		m_XTipMat(i, 2) = m_XMat(i, 2) - m_Ux(0, 2);
	}
	for (int i = 0; i < M; ++i)
	{
		m_YTipMat(i, 0) = m_YMat(i, 0) - m_Uy(0, 0);
		m_YTipMat(i, 1) = m_YMat(i, 1) - m_Uy(0, 1);
		m_YTipMat(i, 2) = m_YMat(i, 2) - m_Uy(0, 2);
	}
}
//==============================================================================

//计算刚性变换矩阵==============================================================
void RigidCPD::ComputeRigidTranMat()
{
	Eigen::MatrixXf A = m_XTipMat.transpose() * m_PMat.transpose() * m_YTipMat;

	Eigen::MatrixXf S = Eigen::MatrixXf::Ones(3, 1);
	Eigen::JacobiSVD<Eigen::Matrix3f> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
	Eigen::MatrixXf U_Mat = svd.matrixU();
	Eigen::MatrixXf V_Mat = svd.matrixV();

	S(2) = (U_Mat * V_Mat.transpose()).determinant();
	m_RotMat = U_Mat * S.asDiagonal() * V_Mat.transpose();
	float AtRTr = (A.transpose() * m_RotMat).trace();

	Eigen::MatrixXf dPM = (m_PMat.rowwise().sum()).asDiagonal();
	Eigen::MatrixXf testM = m_YTipMat.transpose() * dPM * m_YTipMat;
	m_S = AtRTr / (testM.trace() + 1e-12f);
	m_TranMat = m_Ux - m_S * m_Uy * m_RotMat.transpose();

	Eigen::MatrixXf dPN = (m_PMat.colwise().sum()).asDiagonal();
	Eigen::MatrixXf XTrMat = m_XTipMat.transpose() * dPN * m_XTipMat;
	m_Sigma_2 = (XTrMat.trace() - m_S * AtRTr) / (3.0f * m_Np);
	if (m_Sigma_2 <= 0)
		m_Sigma_2 = 1e-8;
}
//==============================================================================

//计算仿射变换矩阵==============================================================
void RigidCPD::ComputeAffineTranMat()
{
	Eigen::MatrixXf A = m_XTipMat.transpose() * m_PMat.transpose() * m_YTipMat;
	Eigen::MatrixXf dPM = (m_PMat.rowwise().sum()).asDiagonal();
	Eigen::MatrixXf H = m_YTipMat.transpose() * dPM * m_YTipMat;
	m_RotMat = A * (H.inverse());
	m_TranMat = m_Ux - m_RotMat * m_Uy;

	Eigen::MatrixXf dPN = (m_PMat.colwise().sum()).asDiagonal();
	Eigen::MatrixXf XTrMat = m_XTipMat.transpose() * dPN * m_XTipMat;
	m_Sigma_2 = (XTrMat.trace() - (A * m_RotMat.transpose()).trace()) / (3.0f * m_Np);
}
//==============================================================================

//点云刚性变换==================================================================
void RigidCPD::RigidTransPC()
{
	m_YMat *= (m_RotMat.transpose());
	m_YMat *= m_S;
	for (int m = 0; m < M; ++m)
	{
		m_YMat(m, 0) += m_TranMat(0);
		m_YMat(m, 1) += m_TranMat(1);
		m_YMat(m, 2) += m_TranMat(2);
	}
}
//==============================================================================

//刚性配准======================================================================
void RigidCPD::Match(PC_XYZ &XPC, PC_XYZ &YPC)
{
	InitRigidCompute(XPC, YPC);
	int iter = 0;

	Eigen::Matrix3f rotMat = Eigen::Matrix3f::Identity();
	Eigen::MatrixXf tMat = Eigen::MatrixXf::Zero(1, 3);
	float s = 1.0f;
	while (m_Sigma_2 > 1e-8 && iter < m_MaxIter)
	{
		//E-Step
		ComputeP();

		//M-Step
		ComputeXTipYTip();
		ComputeRigidTranMat();
		RigidTransPC();
		s *= m_S;
		rotMat = m_S * m_RotMat * rotMat;
		tMat = m_S * m_RotMat * tMat + m_TranMat;
		iter++;
	}

	m_RotMat = rotMat;
	m_TranMat = tMat;
	m_S = s;
}
//==============================================================================

//==============================================================================
void RigidCPD::GetResMat(Eigen::MatrixXf &resMat)
{
	resMat = Eigen::MatrixXf::Identity(4, 4);
	resMat.block<3, 3>(0, 0) = m_RotMat;
	resMat.block<3, 1>(0, 3) = m_TranMat.col(0);

	resMat(0, 3) += m_GX.x - m_S * m_GY.x;
	resMat(1, 3) += m_GX.y - m_S * m_GY.y;
	resMat(2, 3) += m_GX.z - m_S * m_GY.z;
}
//==============================================================================

void TxtToPC(string &txtFile, PC_XYZ &pc, float offset)
{
	fstream txtfile(txtFile, ios::in);
	string coord;
	while (getline(txtfile, coord))
	{
		vector<string> strs;
		boost::split(strs, coord, boost::is_any_of(" "));
		float x = atof(strs[0].data()) + offset;
		float y = atof(strs[1].data()) + offset;
		float z = atof(strs[2].data()) + offset;
		pc.push_back({ x, y, z });
	}
}


void TestRigidMatch()
{
	PC_XYZ XPC, YPC;

	string txtFileX = "C:/Users/Administrator/Downloads/CPD—Data/data1/bunny_source.txt";
	string txtFileY = "C:/Users/Administrator/Downloads/CPD—Data/data1/bunny_target.txt";
	TxtToPC(txtFileX, XPC, 0);
	TxtToPC(txtFileY, YPC, 0);

	for (auto &pt : YPC.points)
	{
		pt.x /= 3.0f;
		pt.y /= 3.0f;
		pt.z /= 3.0f;
	}


	float w = 0.2;
	RigidCPD cpdReg(0.2f, 50, 1e-6);
	cpdReg.Match(XPC, YPC);
	Eigen::MatrixXf resMat;
	cpdReg.GetResMat(resMat);

	PC_XYZ outPt;
	pcl::transformPointCloud(YPC, outPt, resMat);

	pcl::visualization::PCLVisualizer viewer;
	//viewer.addCoordinateSystem(10);
	//显示轨迹
	pcl::visualization::PointCloudColorHandlerCustom<P_XYZ> red(XPC.makeShared(), 255, 0, 0); //设置点云颜色
	viewer.addPointCloud(XPC.makeShared(), red, "XPC");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "XPC");

	pcl::visualization::PointCloudColorHandlerCustom<P_XYZ> green(outPt.makeShared(), 0, 255, 0); //设置点云颜色
	viewer.addPointCloud(outPt.makeShared(), green, "outPt");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "outPt");

	//pcl::visualization::PointCloudColorHandlerCustom<P_XYZ> blue(YPC.makeShared(), 0, 0, 255); //设置点云颜色
	//viewer.addPointCloud(YPC.makeShared(), blue, "YPC");
	//viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "YPC");

	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}
}