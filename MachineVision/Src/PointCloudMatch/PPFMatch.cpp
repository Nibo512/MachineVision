#include "../../include/PointCloudMatch/PPFMatch.h"
#include "../../include/PointCloudFile/PC_Filter.h"
#include "../../include/PointCloudFile/PointCloudOpr.h"

//计算PPF特征=========================================================================
void PPFMATCH::ComputePPFFEATRUE(P_XYZ &p1, P_XYZ &p2, P_N &pn1, P_N &pn2, PPFFEATRUE &ppfFEATRUE)
{
	float diff_x = p1.x - p2.x;
	float diff_y = p1.y - p2.y;
	float diff_z = p1.z - p2.z;

	float dist = std::sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z);
	float normal_ = 1.0f / std::max(dist, 1e-8f);
	diff_x *= normal_; 
	diff_y *= normal_; 
	diff_z *= normal_;

	ppfFEATRUE.dist = dist / m_DistStep;

	float n1n2 = pn1.normal_x * pn2.normal_x + pn1.normal_y * pn2.normal_y + pn1.normal_z * pn2.normal_z;
	ppfFEATRUE.ang_N1N2 = acos(std::max(eps_1, std::min(n1n2, eps1))) / m_AngleStep;

	float n1d = pn1.normal_x * diff_x + pn1.normal_y * diff_y + pn1.normal_z * diff_z;
	ppfFEATRUE.ang_N1D = acos(std::max(eps_1, std::min(n1d, eps1))) / m_AngleStep;

	float n2d = pn2.normal_x * diff_x + pn2.normal_y * diff_y + pn2.normal_z * diff_z;
	ppfFEATRUE.ang_N2D = acos(std::max(eps_1, std::min(n2d, eps1))) / m_AngleStep;
}
//====================================================================================

//构建哈希表==========================================================================
void PPFMATCH::CreateHashMap(PPFFEATRUE &ppfFEATRUE, int i, int j, float alpha)
{
	PPFCELL ppfcell;
	ppfcell.ref_alpha = alpha;
	ppfcell.ref_i = i;
	string key = to_string(ppfFEATRUE.dist) + to_string(ppfFEATRUE.ang_N1D) + to_string(ppfFEATRUE.ang_N2D) + to_string(ppfFEATRUE.ang_N1N2);
	if (m_ModelFeatrue.find(key) == m_ModelFeatrue.end())
	{
		vector<PPFCELL> ppfcells;
		ppfcells.push_back(ppfcell);
		m_ModelFeatrue.insert(std::pair<string, vector<PPFCELL>>(key, ppfcells));
	}
	else
	{
		vector<PPFCELL> &ppfcells = m_ModelFeatrue.find(key)->second;
		ppfcells.push_back(ppfcell);
	}
}
//====================================================================================

//罗格里德斯公式======================================================================
void PPFMATCH::RodriguesFormula(P_N &rotAxis, float rotAng, Eigen::Matrix4f &rotMat)
{
	float cosVal = std::cos(rotAng);
	float conVal_ = 1 - cosVal;
	float sinVal = std::sin(rotAng);

	rotMat(0,0) = cosVal + rotAxis.normal_x * rotAxis.normal_x * conVal_;
	rotMat(0,1) = rotAxis.normal_x * rotAxis.normal_y * conVal_ - rotAxis.normal_z * sinVal;
	rotMat(0,2) = rotAxis.normal_x * rotAxis.normal_z * conVal_ + rotAxis.normal_y * sinVal;

	rotMat(1,0) = rotAxis.normal_y * rotAxis.normal_x * conVal_ + rotAxis.normal_z * sinVal;
	rotMat(1,1) = cosVal + rotAxis.normal_y * rotAxis.normal_y * conVal_;
	rotMat(1,2) = rotAxis.normal_y * rotAxis.normal_z * conVal_ - rotAxis.normal_x * sinVal;

	rotMat(2,0) = rotAxis.normal_z * rotAxis.normal_x * conVal_ - rotAxis.normal_y * sinVal;
	rotMat(2,1) = rotAxis.normal_z * rotAxis.normal_y * conVal_ + rotAxis.normal_x * sinVal;
	rotMat(2,2) = cosVal + rotAxis.normal_z * rotAxis.normal_z * conVal_;
}
//====================================================================================

//计算局部转换矩阵====================================================================
void PPFMATCH::ComputeLocTransMat(P_XYZ &ref_p, P_N &ref_pn, Eigen::Matrix4f &transMat)
{
	transMat = Eigen::Matrix4f::Zero();
	float rotAng = std::acosf(ref_pn.normal_x);
	P_N rotAxis(0.0f, ref_pn.normal_z, -ref_pn.normal_y); //旋转轴垂直于x轴与参考点法向量
	if (abs(rotAxis.normal_y) < EPS && abs(ref_pn.normal_z) < EPS)
	{
		rotAxis.normal_y = 1.0f;
		rotAxis.normal_z = 0.0f;
	}
	else
	{
		float norm = 1.0f / std::sqrt(rotAxis.normal_y * rotAxis.normal_y + rotAxis.normal_z * rotAxis.normal_z);
		rotAxis.normal_y *= norm; rotAxis.normal_z *= norm;
	}
	RodriguesFormula(rotAxis, rotAng, transMat);
	transMat(0,3) = -(transMat(0,0) * ref_p.x + transMat(0,1) * ref_p.y + transMat(0,2) * ref_p.z);
	transMat(1,3) = -(transMat(1,0) * ref_p.x + transMat(1,1) * ref_p.y + transMat(1,2) * ref_p.z);
	transMat(2,3) = -(transMat(2,0) * ref_p.x + transMat(2,1) * ref_p.y + transMat(2,2) * ref_p.z);
	transMat(3,3) = 1.0f;
}
//====================================================================================

//计算局部坐标系下的alpha=============================================================
float PPFMATCH::ComputeLocalAlpha(P_XYZ &ref_p, P_N &ref_pn, P_XYZ &p_, Eigen::Matrix4f &transMat)
{
	float y = transMat(1,0) * p_.x + transMat(1,1) * p_.y + transMat(1,2) * p_.z + transMat(1,3);
	float z = transMat(2,0) * p_.x + transMat(2,1) * p_.y + transMat(2,2) * p_.z + transMat(2,3);
	return std::atan2f(z, y);
}
//====================================================================================

//重置投票器==========================================================================
void PPFMATCH::ResetVoteScheme(vector<vector<int>> &VoteScheme)
{
	for (int i = 0; i < VoteScheme.size(); ++i)
	{
		for (int j = 0; j < VoteScheme[i].size(); ++j)
		{
			VoteScheme[i][j] = 0;
		}
	}
}
//====================================================================================

//创建PPF模板=========================================================================
void PPFMATCH::CreatePPFModel(PC_XYZ &modelPC, PC_N &model_n)
{
	int ptNum = modelPC.size();
	P_XYZ *pModelData = modelPC.points.data();
	P_N *pModelNormal = model_n.points.data();
	for (int i = 0; i < ptNum; ++i)
	{
		P_XYZ &p1 = pModelData[i];
		P_N &pn1 = pModelNormal[i];
		Eigen::Matrix4f transMat;
		ComputeLocTransMat(p1, pn1, transMat);

		for (int j = 0; j < ptNum; ++j)
		{
			if (i != j)
			{
				P_XYZ &p2 = pModelData[j];
				P_N &pn2 = pModelNormal[j];
				PPFFEATRUE ppfFEATRUE;
				ComputePPFFEATRUE(p1, p2, pn1, pn2, ppfFEATRUE);			
				float alpha = ComputeLocalAlpha(p1, pn1, p2, transMat);
				if (std::isnan(alpha))
					continue;
				CreateHashMap(ppfFEATRUE, i, j, alpha);
			}
		}
	}
	m_ModelPC = modelPC;
	m_ModelNormal = model_n;
}
//====================================================================================

//计算变换矩阵======================================================================
void PPFMATCH::ComputeTransMat(Eigen::Matrix4f &SToGMat, float alpha, const Eigen::Matrix4f &RToGMat, Eigen::Matrix4f &transMat)
{
	Eigen::Affine3f rotMat = Eigen::Affine3f::Identity();
	rotMat.rotate(Eigen::AngleAxisf(alpha, Eigen::Vector3f::UnitX()));
	transMat = (SToGMat.inverse()) * rotMat * RToGMat;
}
//==================================================================================

//匹配================================================================================
void PPFMATCH::MatchPose(PC_XYZ &testPC, PC_N &testPCN, Eigen::Matrix4f &resTransMat)
{
	size_t p_number = testPC.size();
	P_XYZ *pTestData = testPC.points.data();
	P_N *pTestPCN = testPCN.points.data();
	vector<PPFPose> v_ppfPose;
	v_ppfPose.resize(p_number);
	int numAngles = floor(CV_2PI / m_AngleStep);
#pragma omp parallel for 
	for (int i = 0; i < p_number; ++i)
	{
		//这个地方有必要的话最好对VoteScheme进行幅零处理
		vector<vector<int>> VoteScheme(m_ModelPC.size());
		for (int j = 0; j < VoteScheme.size(); ++j)
			VoteScheme[j].resize(numAngles);
		//ResetVoteScheme(VoteScheme);
		P_XYZ &ref_p = pTestData[i];
		P_N &ref_pn = pTestPCN[i];
		PPFPose pose;
		ComputeLocTransMat(ref_p, ref_pn, pose.transMat);
		for (int j = 0; j < p_number; ++j)
		{
			if (i != j)
			{
				P_XYZ &p_ = pTestData[j];
				P_N &p_n = pTestPCN[j];
				PPFFEATRUE ppfFEATRUE;
				ComputePPFFEATRUE(ref_p, p_, ref_pn, p_n, ppfFEATRUE);
				string key = to_string(ppfFEATRUE.dist) + to_string(ppfFEATRUE.ang_N1D) + to_string(ppfFEATRUE.ang_N2D) + to_string(ppfFEATRUE.ang_N1N2);
				if (m_ModelFeatrue.find(key) != m_ModelFeatrue.end())
				{
					vector<PPFCELL> &ppfCell_v = m_ModelFeatrue.find(key)->second;
					float alpha_ = ComputeLocalAlpha(ref_p, ref_pn, p_, pose.transMat);
					if (!isnan(alpha_))
					{
						for (int k = 0; k < ppfCell_v.size(); ++k)
						{
							float alpha = alpha_ - ppfCell_v[k].ref_alpha;
							if (alpha < 0)
								alpha += CV_2PI;
							if (alpha > CV_2PI)
								alpha -= CV_2PI;
							int alpha_index = floor((alpha) / m_AngleStep);
							VoteScheme[ppfCell_v[k].ref_i][alpha_index]++;
						}
					}
				}
			}
		}
		pose.votes = VoteScheme[0][0];
		for (int j = 0; j < VoteScheme.size(); ++j)
		{
			for (int k = 0; k < numAngles; ++k)
			{
				if (pose.votes < VoteScheme[j][k])
				{
					pose.votes = VoteScheme[j][k];
					pose.ref_i = j; pose.i_ = k;
				}
			}
		}
		v_ppfPose[i] = pose;
	}
	int poseIdx = 0, poseVote = 0;
	for (int i = 0; i < v_ppfPose.size(); ++i)
	{
		if (poseVote < v_ppfPose[i].votes)
		{
			poseVote = v_ppfPose[i].votes;
			poseIdx = i;
		}
	}
	PPFPose &pose = v_ppfPose[poseIdx];
	Eigen::Matrix4f modelPose;
	ComputeLocTransMat(m_ModelPC[pose.ref_i], m_ModelNormal[pose.ref_i], modelPose);

	float alpha = pose.i_ * m_AngleStep;
	ComputeTransMat(pose.transMat, alpha, modelPose, resTransMat);
}
//====================================================================================

////排序==============================================================================
//bool ComparePose(PPFPose& a, PPFPose& b)
//{
//	return a.votes > b.votes;
//}
////==================================================================================

////判定条件==========================================================================
//bool DecisionCondition(PPFPose& a, PPFPose& b, float angThres, float distThres)
//{
//	float angle_a = ComputeRotMatAng(a.transMat);
//	float angle_b = ComputeRotMatAng(b.transMat);
//
//	float* pATransMat = a.transMat.ptr<float>();
//	float* pBTransMat = a.transMat.ptr<float>();
//	float diff_x = pATransMat[3] - pBTransMat[3];
//	float diff_y = pATransMat[7] - pBTransMat[7];
//	float diff_z = pATransMat[11] - pBTransMat[11];
//	
//	return (abs(angle_a - angle_b) < angThres && (diff_x * diff_x + diff_y * diff_y + diff_z * diff_z) < distThres);
//}
////==================================================================================

////非极大值抑制======================================================================
//void NonMaxSuppression(vector<PPFPose>& ppfPoses, vector<PPFPose>& resPoses, float angThres, float distThres)
//{
//	std::sort(ppfPoses.begin(), ppfPoses.end(), ComparePose);
//	size_t pose_num = ppfPoses.size();
//	vector<bool> isLabel(pose_num, false);
//	for (size_t i = 0; i < pose_num; ++i)
//	{
//		if (isLabel[i])
//			continue;
//		PPFPose& ref_pose = ppfPoses[i];
//		resPoses.push_back(ref_pose);
//		isLabel[i] = true;
//		for (size_t j = i; j < pose_num; ++j)
//		{
//			if (!isLabel[j])
//			{
//				PPFPose& pose_ = ppfPoses[i];
//				if (DecisionCondition(ref_pose, pose_, angThres, distThres))
//				{
//					isLabel[j] = true;
//				}
//			}
//		}
//	}
//}
////==================================================================================


void CreateTransMat(Eigen::Affine3f &transform)
{
	transform = Eigen::Affine3f::Identity();
	transform.translation() << 56, 15, 95;

	Eigen::Affine3f transformX = Eigen::Affine3f::Identity();
	transformX.rotate(Eigen::AngleAxisf(2.2, Eigen::Vector3f::UnitX()));
	Eigen::Affine3f transformY = Eigen::Affine3f::Identity();
	transformY.rotate(Eigen::AngleAxisf(0.3, Eigen::Vector3f::UnitY()));
	Eigen::Affine3f transformZ = Eigen::Affine3f::Identity();
	transformZ.rotate(Eigen::AngleAxisf(1.9, Eigen::Vector3f::UnitZ()));
	transform = transform * transformX * transformY *transformZ;
}

//测试程序==========================================================================
void PPFTestProgram()
{
	//创建PPF模板===========================
	PC_XYZ modelPC;
	pcl::io::loadPCDFile("H:/pcl-learning-master/14registration配准/5刚性物体的鲁棒姿态估计/chef.pcd",modelPC);

	//计算降采样大小
	P_XYZ min_p, max_p;
	pcl::getMinMax3D(modelPC, min_p, max_p);
	float stepSample = 0.03 * (max_p.x - min_p.x);

	//计算法向量
	P_XYZ viewPt(0, 0, 1000);
	PC_XYZ downModelPC;
	//PC_VoxelGrid(modelPC, downModelPC, stepSample);
	PC_N modelNormal;
	PC_ComputePCNormal(downModelPC, modelNormal, 2 * stepSample);
	
	//创建模板
	PPFMATCH ppfModel(2.0f / 180 * CV_PI, stepSample);
	ppfModel.CreatePPFModel(downModelPC, modelNormal);
	//========================================


	
	//配准部分================================
	Eigen::Affine3f transform;
	CreateTransMat(transform);
	PC_XYZ testPC;
	pcl::io::loadPCDFile("H:/pcl-learning-master/14registration配准/5刚性物体的鲁棒姿态估计/rs1.pcd", testPC);
	PC_XYZ downTestPC;
	PC_VoxelGrid(testPC, downTestPC, stepSample);
	//PC_XYZ downTestPC_T;
	//pcl::transformPointCloud(downTestPC, downTestPC_T, transform);
	//P_XYZ viewPt_T = pcl::transformPoint(viewPt, transform);

	PC_N testNormal;
	PC_ComputePCNormal(downTestPC, testNormal, 2 * stepSample);

	Eigen::Matrix4f resTransMat;
	ppfModel.MatchPose(downTestPC, testNormal, resTransMat);
	//========================================

	PC_XYZ resPC;
	pcl::transformPointCloud(modelPC, resPC, resTransMat);

	pcl::visualization::PCLVisualizer viewer;
	//viewer.addCoordinateSystem(10);
	//pcl::visualization::PointCloudColorHandlerCustom<P_XYZ> white(downModelPC.makeShared(), 255, 255, 255);
	//viewer.addPointCloud(downModelPC.makeShared(), white, "downModelPC");
	//viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "downModelPC");

	pcl::visualization::PointCloudColorHandlerCustom<P_XYZ> red(testPC.makeShared() , 255, 0, 0);
	viewer.addPointCloud(testPC.makeShared(), red, "testPC");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "testPC");

	pcl::visualization::PointCloudColorHandlerCustom<P_XYZ> green(resPC.makeShared(), 0, 255, 0);
	viewer.addPointCloud(resPC.makeShared(), green, "resPC");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "resPC");

	//pcl::visualization::PointCloudColorHandlerCustom<P_XYZ> blue(downModelPC.makeShared(), 0, 0, 255);
	//viewer.addPointCloud(downModelPC.makeShared(), blue, "downModelPC");
	//viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "downModelPC");

	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}
}
//==================================================================================