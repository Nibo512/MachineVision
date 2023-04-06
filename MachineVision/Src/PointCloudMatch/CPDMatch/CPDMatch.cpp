#include "../../../include/PointCloudMatch/CPDMatch/CPDMatch.h"
#include <Eigen/Eigenvalues>
#include "../../../include/PointCloudFile/PointCloudOpr.h"

//计算点云的缩放参数============================================================
void CPD::GetScaleParam(PC_XYZ &XPC, PC_XYZ &YPC)
{
	P_XYZ min_px, max_px, min_py, max_py;
	pcl::getMinMax3D(XPC, min_px, max_px);
	pcl::getMinMax3D(YPC, min_py, max_py);
	m_ScaleX = abs(max_px.x - min_px.x) > abs(max_py.x - min_py.x) ? 
		abs(max_px.x - min_px.x) : abs(max_py.x - min_py.x);
	m_ScaleY = abs(max_px.y - min_px.y) > abs(max_py.y - min_py.y) ? 
		abs(max_px.y - min_px.y) : abs(max_py.y - min_py.y);
	m_ScaleZ = abs(max_px.z - min_px.z) > abs(max_py.z - min_py.z) ?
		abs(max_px.z - min_px.z) : abs(max_py.z - min_py.z);

	m_ScaleX += 1e-12;	m_ScaleY += 1e-12; m_ScaleZ += 1e-12;
}
//==============================================================================

//点云去中心化==================================================================
void CPD::PCCentralization(PC_XYZ &XPC, PC_XYZ &YPC)
{
	PC_GetPCGravity(XPC, m_GX);
	PC_GetPCGravity(YPC, m_GY);
	PCToMat(XPC, m_XMat, m_GX);
	PCToMat(YPC, m_YMat, m_GY);
}
//==============================================================================

//数据转化======================================================================
void CPD::PCToMat(PC_XYZ &pc, Eigen::MatrixXf &mat, P_XYZ &gravity)
{
	int num = pc.size();
	mat = Eigen::MatrixXf::Zero(num, 3);
	P_XYZ *pPCData = pc.points.data();
	for (int i = 0; i < num; ++i)
	{
		mat(i, 0) = (pPCData[i].x - gravity.x) / m_ScaleX;
		mat(i, 1) = (pPCData[i].y - gravity.y) / m_ScaleY;
		mat(i, 2) = (pPCData[i].z - gravity.z) / m_ScaleZ;
	}
}
//==============================================================================

//初始化sigma===================================================================
void CPD::InitSigma(PC_XYZ &XPC, PC_XYZ &YPC)
{
	m_Sigma_2 = 0.0f;
	for (int n = 0; n < N; ++n)
	{
		P_XYZ &X_ = XPC[n];
		for (int m = 0; m < M; ++m)
		{
			P_XYZ &Y_ = YPC[m];
			float diff_x = X_.x - Y_.x;
			float diff_y = X_.y - Y_.y;
			float diff_z = X_.z - Y_.z;
			m_Sigma_2 += diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;
		}
	}
	m_Sigma_2 /= (3.0f * M * N);
}
//==============================================================================

//计算P=========================================================================
void CPD::ComputeP()
{
	float res = std::pow(CV_2PI * m_Sigma_2, 1.5f) * m_W / (1.0 - m_W) * float(M) / float(N);
	float sigma_2 = -1.0f / (2.0f * m_Sigma_2);
#pragma omp parallel for
	for (int n = 0; n < N; ++n)
	{
		float sum_ = 0.0f;
		float x_x = m_XMat(n, 0);
		float x_y = m_XMat(n, 1);
		float x_z = m_XMat(n, 2);
		for (int m = 0; m < M; ++m)
		{
			float diff_x = x_x - m_YMat(m, 0);
			float diff_y = x_y - m_YMat(m, 1);
			float diff_z = x_z - m_YMat(m, 2);
			m_PMat(m, n) = (diff_x * diff_x + diff_y * diff_y + diff_z * diff_z) * sigma_2;
			m_PMat(m, n) = std::exp(m_PMat(m, n));
			sum_ += m_PMat(m, n);
		}
		sum_ += res;
		for (int m = 0; m < M; ++m)
		{
			m_PMat(m, n) /= (sum_ + 1e-12);
		}
	}
}
//==============================================================================

//初始化计算====================================================================
void CPD::InitCompute(PC_XYZ &XPC, PC_XYZ &YPC)
{
	M = YPC.size(); N = XPC.size();
	InitSigma(XPC, YPC);
	GetScaleParam(XPC, YPC);
	PCCentralization(XPC, YPC);
	m_PMat = Eigen::MatrixXf::Zero(M, N);
}
//==============================================================================
