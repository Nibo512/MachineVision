#include "../../include/PointCloudMatch/PLIcpMatch.h"
#include "../../include/PointCloudFile/PC_Filter.h"
#include "../../include/PointCloudFile/PC_Filter.h"

//最小二乘法拟合空间直线==========================================================
void JC_Fit3DLineIdx_(PC_XYZ &pts, vector<int> &ptIdx, vector<float> &line)
{
	if (line.size() != 6)
		line.resize(6);
	float w_x_sum = 0.0f, w_y_sum = 0.0f, w_z_sum = 0.0f;
	float w_xy_sum = 0.0f, w_yz_sum = 0.0f, w_zx_sum = 0.0f;
	P_XYZ *pPt = pts.points.data();
	int ptNum = ptIdx.size();
	int *pIdx = ptIdx.data();
	for (int i = 0; i < ptNum; ++i)
	{
		w_x_sum += pPt[pIdx[i]].x;
		w_y_sum += pPt[pIdx[i]].y;
		w_z_sum += pPt[pIdx[i]].z;
	}
	float w_sum = 1.0f / max((float)ptNum, 1e-8f);
	float w_x_mean = w_x_sum * w_sum;
	float w_y_mean = w_y_sum * w_sum;
	float w_z_mean = w_z_sum * w_sum;

	cv::Mat A(3, 3, CV_32FC1, cv::Scalar(0));
	float* pA = A.ptr<float>(0);
	for (int i = 0; i < ptNum; ++i)
	{
		float x = pPt[pIdx[i]].x, y = pPt[pIdx[i]].y, z = pPt[pIdx[i]].z;
		float x_ = x - w_x_mean;
		float y_ = y - w_y_mean;
		float z_ = z - w_z_mean;

		pA[0] += (y_ * y_ + z_ * z_);
		pA[1] -= x_ * y_;
		pA[2] -= z_ * x_;
		pA[4] += (x_ * x_ + z_ * z_);
		pA[5] -= y_ * z_;
		pA[8] += (x_ * x_ + y_ * y_);
	}
	pA[3] = pA[1]; pA[6] = pA[2]; pA[7] = pA[5];

	cv::Mat eigenVal, eigenVec;
	cv::eigen(A, eigenVal, eigenVec);
	float* pEigenVec = eigenVec.ptr<float>(2);
	line[0] = pEigenVec[0]; line[1] = pEigenVec[1]; line[2] = pEigenVec[2];
	line[3] = w_x_mean; line[4] = w_y_mean; line[5] = w_z_mean;
}
//================================================================================

void ComputePtLine(PC_XYZ &srcPC, PC_N &normals, int size)
{
	int ptNum = srcPC.size();
	normals.resize(ptNum);
	KdTreeFLANN<P_XYZ> kdtree;
	kdtree.setInputCloud(srcPC.makeShared());
	vector<int> PIdx(size);
	vector<float> PDist(size);
	vector<float> line(6);
	for (int i = 0; i < ptNum; ++i)
	{
		P_XYZ &pt = srcPC[i];
		kdtree.nearestKSearch(pt, size, PIdx, PDist);
		JC_Fit3DLineIdx_(srcPC, PIdx, line);
		normals[i] = { line[0], line[1], line[2] };
	}
}

//寻找最邻近点====================================================================
void JCMATCH::JC_FindKnnPair()
{
	int ptNum = m_SrcPC.size();
	P_XYZ *pSrc = m_SrcPC.points.data();
	m_PairIdxes.resize(0);
	m_PairIdxes.reserve(ptNum);

	for (int i = 0; i < ptNum; ++i)
	{
		vector<int> PIdx;
		vector<float> PDist;
		m_TgtKdTree.nearestKSearch(pSrc[i], 1, PIdx, PDist);
		if (PDist[0] < m_MaxPPDist)
		{
			m_PairIdxes.push_back(PairIdx(i, PIdx[0]));
		}
	}
}
//================================================================================

//计算变换矩阵---仿射变换=========================================================
void JCMATCH::JC_ComputeAffineTransMat(Eigen::Matrix4f &transMat)
{
	int ptNum = m_PairIdxes.size();
	P_XYZ *pSrc = m_SrcPC.points.data();
	P_XYZ *pTgt = m_TgtPC.points.data();
	PairIdx *pPairIdx = m_PairIdxes.data();
	Eigen::Vector3f sum(0.0f, 0.0f, 0.0f), sum_t(0.0f, 0.0f, 0.0f);
	for (int i = 0; i < ptNum; ++i)
	{
		P_XYZ &srcp = pSrc[pPairIdx[i].srcIdx];
		P_XYZ &tgtp = pTgt[pPairIdx[i].tgtIdx];
		sum[0] += srcp.x; sum[1] += srcp.y; sum[2] += srcp.z;
		sum_t[0] += tgtp.x; sum_t[1] += tgtp.y; sum_t[2] += tgtp.z;
	}
	Eigen::Vector3f mean = sum / (ptNum + 1e-8), mean_t = sum_t / (ptNum + 1e-8);

	Eigen::Matrix3f A = Eigen::Matrix3f::Zero();
	Eigen::Matrix3f B = Eigen::Matrix3f::Zero();
	for (int i = 0; i < ptNum; ++i)
	{
		int srcIdx = pPairIdx[i].srcIdx;
		float x_ = (pSrc[srcIdx].x - mean[0]);
		float y_ = (pSrc[srcIdx].y - mean[1]);
		float z_ = (pSrc[srcIdx].z - mean[2]);

		int tgtIdx = pPairIdx[i].tgtIdx;
		float tx_ = (pTgt[tgtIdx].x - mean_t[0]);
		float ty_ = (pTgt[tgtIdx].y - mean_t[1]);
		float tz_ = (pTgt[tgtIdx].z - mean_t[2]);

		A(0, 0) += x_ * x_; A(0, 1) += x_ * y_; A(0, 2) += x_ * z_;
		A(1, 1) += y_ * y_; A(1, 2) += y_ * z_; A(2, 2) += z_ * z_;

		B(0, 0) += x_ * tx_; B(1, 0) += y_ * tx_; B(2, 0) += z_ * tx_;
		B(0, 1) += x_ * ty_; B(1, 1) += y_ * ty_; B(2, 1) += z_ * ty_;
		B(0, 2) += x_ * tz_; B(1, 2) += y_ * tz_; B(2, 2) += z_ * tz_;
	}
	A(1, 0) = A(0, 1); A(2, 0) = A(0, 2); A(2, 1) = A(1, 2);
	transMat.block<3, 3>(0, 0) = (A.inverse() * B).transpose();
	transMat.block<3, 1>(0, 3) = mean_t - transMat.block<3, 3>(0, 0) * mean;
}
//================================================================================

//刚性变换========================================================================
void JCMATCH::JC_ComputeRigidTranMat(Eigen::Matrix4f &transMat)
{
	int ptNum = m_PairIdxes.size();
	P_XYZ *pSrc = m_SrcPC.points.data();
	P_XYZ *pTgt = m_TgtPC.points.data();
	PairIdx *pPairIdx = m_PairIdxes.data();

	Eigen::Vector3f sum(0.0f, 0.0f, 0.0f), sum_t(0.0f, 0.0f, 0.0f),
		mean(0.0f, 0.0f, 0.0f), mean_t(0.0f, 0.0f, 0.0f);
	double sum_w = 0.0;
	for (int i = 0; i < ptNum; ++i)
	{
		P_XYZ &srcp = pSrc[pPairIdx[i].srcIdx];
		P_XYZ &tgtp = pTgt[pPairIdx[i].tgtIdx];
		sum[0] += srcp.x; sum[1] += srcp.y; sum[2] += srcp.z;
		sum_t[0] += tgtp.x; sum_t[1] += tgtp.y; sum_t[2] += tgtp.z;
	}
	mean = sum / (ptNum + 1e-8); mean_t = sum_t / (ptNum + 1e-8);

	Eigen::Matrix3f sigma = Eigen::Matrix3f::Zero();
	for (int i = 0; i < ptNum; ++i)
	{
		int srcIdx = pPairIdx[i].srcIdx;
		float x_ = (pSrc[srcIdx].x - mean[0]);
		float y_ = (pSrc[srcIdx].y - mean[1]);
		float z_ = (pSrc[srcIdx].z - mean[2]);

		int tgtIdx = pPairIdx[i].tgtIdx;
		float tx_ = (pTgt[tgtIdx].x - mean_t[0]);
		float ty_ = (pTgt[tgtIdx].y - mean_t[1]);
		float tz_ = (pTgt[tgtIdx].z - mean_t[2]);

		sigma(0, 0) += x_ * tx_; sigma(0, 1) += y_ * tx_; sigma(0, 2) += z_ * tx_;
		sigma(1, 0) += x_ * ty_; sigma(1, 1) += y_ * ty_; sigma(1, 2) += z_ * ty_;
		sigma(2, 0) += x_ * tz_; sigma(2, 1) += y_ * tz_; sigma(2, 2) += z_ * tz_;
	}
	
	Eigen::MatrixXf S = Eigen::MatrixXf::Ones(3, 1);
	Eigen::JacobiSVD<Eigen::Matrix3f> svd(sigma, Eigen::ComputeFullV | Eigen::ComputeFullU);
	Eigen::MatrixXf U_Mat = svd.matrixU();
	Eigen::MatrixXf V_Mat = svd.matrixV();
	if (U_Mat.determinant() * V_Mat.determinant() < 0)
		S(2) = -1;
	transMat.block<3, 3>(0, 0) = U_Mat * S.asDiagonal() * V_Mat.transpose();
	transMat.block<3, 1>(0, 3) = mean_t - transMat.block<3, 3>(0, 0) * mean;
}
//================================================================================

//计算匹配分数====================================================================
float JCMATCH::ComputeFitScore()
{
	return (float)m_EffPtNum / (float)m_SrcPC.size();
}
//================================================================================

//计算loss========================================================================
bool JCMATCH::ComputeLoss()
{
	int ptNum = m_SrcPC.size();
	P_XYZ *pSrc = m_SrcPC.points.data();
	float loss = 0.0f;
	m_EffPtNum = 0;
	vector<int> PIdx(1);
	vector<float> PDist(1);
	for (int i = 0; i < ptNum; ++i)
	{
		m_TgtKdTree.nearestKSearch(pSrc[i], 1, PIdx, PDist);
		if (PDist[0] < m_MaxPPDist)
		{
			loss += PDist[0];
			m_EffPtNum++;
		}
	}
	loss /= m_EffPtNum;
	if (abs(loss - m_Loss) < m_Eps)
	{
		m_Loss = loss;
		return true;
	}
	else
	{
		m_Loss = loss;
		return false;
	}
}
//================================================================================

//构建转换矩阵====================================================================
void ConstructTransMat(double alpha, double beta, double gamma,
	double tx, double ty, double tz, Eigen::Matrix4f &tranMat)
{
	tranMat = Eigen::Matrix4f::Zero();
	tranMat(0, 0) = cos(gamma) * cos(beta);
	tranMat(0, 1) = -sin(gamma) * cos(alpha) + cos(gamma) * sin(beta) * sin(alpha);
	tranMat(0, 2) = sin(gamma) * sin(alpha) + cos(gamma) * sin(beta) * cos(alpha);
	tranMat(1, 0) = sin(gamma) * cos(beta);
	tranMat(1, 1) = cos(gamma) * cos(alpha) + sin(gamma) * sin(beta) * sin(alpha);
	tranMat(1, 2) = -cos(gamma) * sin(alpha) + sin(gamma) * sin(beta) * cos(alpha);
	tranMat(2, 0) = -sin(beta);
	tranMat(2, 1) = cos(beta) * sin(alpha);
	tranMat(2, 2) = cos(beta) * cos(alpha);

	tranMat(0, 3) = tx; tranMat(1, 3) = ty; tranMat(2, 3) = tz; tranMat(3, 3) = 1.0f;
}
//================================================================================

//带法向量的======================================================================
void JCMATCH::JC_ComputeTransMatWithNormal(Eigen::Matrix4f &transMat)
{
	Eigen::Matrix<double, 6, 6> ATA = Eigen::Matrix<double, 6, 6>::Zero();
	Eigen::Matrix<double, 6, 1> ATb = Eigen::Matrix<double, 6, 1>::Zero();

	int ptNum = m_PairIdxes.size();
	P_XYZ *pSrc = m_SrcPC.points.data();
	P_XYZ *pTgt = m_TgtPC.points.data();
	P_N *pTpn = m_TgtPtVecs.points.data();
	PairIdx *pPairIdx = m_PairIdxes.data();
	for (int i = 0; i < ptNum; ++i)
	{
		const P_XYZ &spt = pSrc[pPairIdx[i].srcIdx];
		const P_XYZ &tpt = pTgt[pPairIdx[i].tgtIdx];
		const P_N &tpn = pTpn[pPairIdx[i].tgtIdx];

		double a = tpn.normal_z * spt.y - tpn.normal_y * spt.z;
		double b = tpn.normal_x * spt.z - tpn.normal_z * spt.x;
		double c = tpn.normal_y * spt.x - tpn.normal_x * spt.y;

		ATA(0, 0) += a * a; ATA(0, 1) += a * b; ATA(0, 2) += a * c;
		ATA(0, 3) += a * tpn.normal_x; ATA(0, 4) += a * tpn.normal_y; ATA(0, 5) += a * tpn.normal_z;

		ATA(1, 1) += b * b; ATA(1, 2) += b * c; 
		ATA(1, 3) += b * tpn.normal_x; ATA(1, 4) += b * tpn.normal_y; ATA(1, 5) += b * tpn.normal_z;

		ATA(2, 2) += c * c; 
		ATA(2, 3) += c * tpn.normal_x; ATA(2, 4) += c * tpn.normal_y; ATA(2, 5) += c * tpn.normal_z;

		ATA(3, 3) += tpn.normal_x * tpn.normal_x; ATA(3, 4) += tpn.normal_x * tpn.normal_y; ATA(3, 5) += tpn.normal_x * tpn.normal_z;
		ATA(4, 4) += tpn.normal_y * tpn.normal_y; ATA(4, 5) += tpn.normal_y * tpn.normal_z;
		ATA(5, 5) += tpn.normal_z * tpn.normal_z;

		double d = (tpt.x - spt.x) * tpn.normal_x + (tpt.y - spt.y) * tpn.normal_y + (tpt.z - spt.z) * tpn.normal_z;
		ATb(0, 0) += a * d; ATb(1, 0) += b * d; ATb(2, 0) += c * d;
		ATb(3, 0) += tpn.normal_x * d; ATb(4, 0) += tpn.normal_y * d; ATb(5, 0) += tpn.normal_z * d;
	}

	ATA(1, 0) = ATA(0, 1);
	ATA(2, 0) = ATA(0, 2); ATA(2, 1) = ATA(1, 2);
	ATA(3, 0) = ATA(0, 3); ATA(3, 1) = ATA(1, 3); ATA(3, 2) = ATA(2, 3);
	ATA(4, 0) = ATA(0, 4); ATA(4, 1) = ATA(1, 4); ATA(4, 2) = ATA(2, 4); ATA(4, 3) = ATA(3, 4);
	ATA(5, 0) = ATA(0, 5); ATA(5, 1) = ATA(1, 5); ATA(5, 2) = ATA(2, 5); ATA(5, 3) = ATA(3, 5); ATA(5, 4) = ATA(4, 5);

	Eigen::Matrix<double, 6, 1> x = static_cast<Eigen::Matrix<double, 6, 1>> (ATA.inverse() * ATb);
	ConstructTransMat(x(0), x(1), x(2), x(3), x(4), x(5), transMat);
}
//================================================================================

//匹配程序========================================================================
float JCMATCH::JC_PCMatch(PC_XYZ &srcPC, PC_XYZ &tgtPC, Eigen::Matrix4f &transMat)
{
	//计算目标点云中没点的方向向量
	m_TgtPC = tgtPC;
	ComputePtLine(m_TgtPC, m_TgtPtVecs, 5);
	//CalPCNormal(m_TgtPC, m_TgtPtVecs, 3.0f);
	m_TgtKdTree.setInputCloud(m_TgtPC.makeShared());
	pcl::transformPointCloud(srcPC, m_SrcPC, transMat);
	Eigen::Matrix4f transMat_ = Eigen::Matrix4f::Identity();
	for (int k = 0; k < m_MaxIter; ++k)
	{
		//获取配对点
		JC_FindKnnPair();
		if (m_PairIdxes.size() < 4)
			return 0;
		//给原始点云增加权重
		JC_ComputeTransMatWithNormal(transMat_);
		for (int idx = 0; idx < 12; ++idx)
		{
			if (isnan(transMat_(idx)))
				return 0;
		}
		transMat = transMat_ * transMat;
		pcl::transformPointCloud(srcPC, m_SrcPC, transMat);
		if (ComputeLoss())
		{
			break;
		}
	}
	return ComputeFitScore();
}
//================================================================================

void PLIcpMatchTest()
{
	PC_XYZ processSample, modelSample;
	pcl::io::loadPLYFile("D:/熟悉软件和算法用/smoothTrack.ply", processSample);
	pcl::io::loadPLYFile("D:/熟悉软件和算法用/m_PolishTrack.ply", modelSample);
	pcl::io::loadPLYFile("D:/熟悉软件和算法用/m_PolishTrack.ply", modelSample);

	//PC_XYZ voxelPC;
	//PC_VoxelGrid(modelSample, voxelPC, 2.0);

	JCMATCH JCMATCHTest;

	Eigen::Matrix4f transMat = Eigen::Matrix4f::Identity();
	JCMATCHTest.JC_PCMatch(processSample, modelSample, transMat);

	PC_XYZ smoothTrack_T;
	pcl::transformPointCloud(processSample, smoothTrack_T, transMat);

	pcl::visualization::PCLVisualizer viewer;
	viewer.addCoordinateSystem(10);
	//显示轨迹
	pcl::visualization::PointCloudColorHandlerCustom<P_XYZ> red(processSample.makeShared(), 255, 0, 0); //设置点云颜色
	viewer.addPointCloud(processSample.makeShared(), red, "processSample");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "processSample");

	pcl::visualization::PointCloudColorHandlerCustom<P_XYZ> green(modelSample.makeShared(), 0, 255, 0); //设置点云颜色
	viewer.addPointCloud(modelSample.makeShared(), green, "modelSample");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "modelSample");

	pcl::visualization::PointCloudColorHandlerCustom<P_XYZ> blue(smoothTrack_T.makeShared(), 0, 0, 255); //设置点云颜色
	viewer.addPointCloud(smoothTrack_T.makeShared(), blue, "smoothTrack_T");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "smoothTrack_T");

	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}
}