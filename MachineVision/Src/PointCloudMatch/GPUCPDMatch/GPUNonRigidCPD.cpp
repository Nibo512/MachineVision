#include "../../../include/PointCloudMatch/GPUCPDMatch/GPUNonRigidCPD.h"
#include "../../../include/PointCloudMatch/GPUCPDMatch/GPUCPDMatch.cuh"
#include "../../../include/PointCloudFile/PC_Filter.h"
#include "../../../include/PointCloudFile/PointCloudOpr.h"

//计算P矩阵=====================================================================
void GPUNonRigidCPD::ComputetPMat()
{
	float res = std::pow(CV_2PI * m_Sigma_2, 1.5f) * m_W / (1.0 - m_W) * float(M) / float(N);
	float sigma_2 = -1.0f / (2.0f * m_Sigma_2);
#pragma omp parallel for
	for (int n = 0; n < N; ++n)
	{
		float sum_ = 0.0f;
		float x_x = m_XPC[n].x;
		float x_y = m_XPC[n].y;
		float x_z = m_XPC[n].z;
		for (int m = 0; m < M; ++m)
		{
			float diff_x = x_x - m_YPC[m].x;
			float diff_y = x_y - m_YPC[m].y;
			float diff_z = x_z - m_YPC[m].z;
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

//==============================================================================
GPUNonRigidCPD::~GPUNonRigidCPD()
{
	if (m_pGMat_H != nullptr)
	{
		delete[] m_pGMat_H;
		m_pGMat_H = nullptr;
	}
	if (m_GDiagMat != nullptr)
	{
		delete[] m_GDiagMat;
		m_GDiagMat = nullptr;
	}
}
//==============================================================================

//点云去中心化==================================================================
void GPUNonRigidCPD::PCCentralization(PC_XYZ& XPC, PC_XYZ& YPC)
{
	PC_GetPCGravity(XPC, m_GX);
	PC_GetPCGravity(YPC, m_GY);

	for (P_XYZ& pt : XPC.points)
	{
		pt.x = (pt.x - m_GX.x) / m_ScaleX;
		pt.y = (pt.y - m_GX.y) / m_ScaleY;
		pt.z = (pt.z - m_GX.z) / m_ScaleZ;
	}
	for (P_XYZ& pt : YPC.points)
	{
		pt.x = (pt.x - m_GX.x) / m_ScaleX;
		pt.y = (pt.y - m_GX.y) / m_ScaleY;
		pt.z = (pt.z - m_GX.z) / m_ScaleZ;
	}
}
//==============================================================================

//初始化计算====================================================================
void GPUNonRigidCPD::InitGPUCompute(PC_XYZ& XPC, PC_XYZ& YPC)
{
	m_Beta = 2.0f;
	m_YPC = YPC;
	m_XPC = XPC;
	M = YPC.size(); N = XPC.size();
	InitSigma(m_XPC, m_YPC);
	GetScaleParam(m_XPC, m_YPC);
	PCCentralization(m_XPC, m_YPC);
	m_pGMat_H = new float[M * M];
	ConstructGMat();
}
//==============================================================================

//计算G矩阵=====================================================================
void GPUNonRigidCPD::ConstructGMat()
{
	float beta2 = -1.0f / (2.0f * m_Beta);
	for (int m1 = 0; m1 < M; ++m1)
	{
		int off_m1 = M * m1;
		for (int m2 = 0; m2 <= m1; ++m2)
		{
			if (m2 == m1)
				m_pGMat_H[off_m1 + m2] = 1.0f;
			else
			{
				float diff_x = m_YPC[m1].x - m_YPC[m2].x;
				float diff_y = m_YPC[m1].y - m_YPC[m2].y;
				float diff_z = m_YPC[m1].z - m_YPC[m2].z;
				m_pGMat_H[off_m1 + m2] = std::exp((diff_x * diff_x + diff_y * diff_y + diff_z * diff_z) * beta2);
				m_pGMat_H[m2 * M + m1] = m_pGMat_H[off_m1 + m2];
			}
		}
	}

	//m_GDiagMat = new float[M];
	//CalGMatInvKernel(m_pGMat_H, M, m_GDiagMat);

	//fstream eigenFile("D:/GPUInv.csv", ios::app);
	//for (int m1 = M - 1; m1 > -1; --m1)
	//{
	//	eigenFile << m_GDiagMat[m1] << ",";
	//}
	//eigenFile.close();
} 
//==============================================================================

//计算A=========================================================================
//void GPUNonRigidCPD::GPUComputeA(Eigen::MatrixXf& A, Eigen::MatrixXf& dPM, float c)
//{
//
//}
//==============================================================================

//非刚性配准====================================================================
void GPUNonRigidCPD::Match(PC_XYZ& XPC, PC_XYZ& YPC)
{
	InitGPUCompute(XPC, YPC);
}
//==============================================================================

void TestGPUNonRigidMatch()
{
	PC_XYZ XPC_, YPC_;
	pcl::io::loadPLYFile("D:/data/变形测试点云/EV/3.ply", YPC_);
	pcl::io::loadPLYFile("D:/data/变形测试点云/EV/2.ply", XPC_);

	PC_XYZ XPC, YPC;
	PC_VoxelGrid(XPC_, XPC, 8.0);
	PC_VoxelGrid(YPC_, YPC, 8.0);

	float w = 0.2;
	GPUNonRigidCPD cpdReg(w, 50, 1e-8f);
	cpdReg.Match(XPC, YPC);
}
