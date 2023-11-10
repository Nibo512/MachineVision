#include "../../include/PointCloudFile/PC_Filter.h"

//体素滤波===========================================================================
void PC_VoxelGrid(PC_XYZ& srcPC, PC_XYZ& dstPC, float leafSize)
{
	if (srcPC.size() == 0)
		return;
	VoxelGrid<P_XYZ> vg;
	vg.setInputCloud(srcPC.makeShared());
	vg.setLeafSize(leafSize, leafSize, leafSize);
	vg.filter(dstPC);
}
//===================================================================================

//直通滤波===========================================================================
void PC_PassFilter(PC_XYZ &srcPC, PC_XYZ &dstPC, const string mode, double minVal, double maxVal)
{
	if (srcPC.empty())
		return;
	if (mode != "x" && mode != "y" && mode != "z")
		return;
	PassThrough<P_XYZ> pt;
	pt.setInputCloud(srcPC.makeShared());
	pt.setFilterFieldName(mode);
	pt.setFilterLimits(minVal, maxVal);
	pt.filter(dstPC);
}
//===================================================================================

//导向滤波===========================================================================
void PC_GuideFilter(PC_XYZ& srcPC, PC_XYZ& dstPC, int radius, double lamda)
{
	KdTreeFLANN<P_XYZ> kdtree;
	kdtree.setInputCloud(srcPC.makeShared());
	int pts_num = srcPC.size();
	dstPC.resize(pts_num);
	for (int i = 0; i < pts_num; ++i)
	{
		vector<int> PIdx;
		vector<float> PDist;
		P_XYZ& src_p = srcPC[i];
		kdtree.nearestKSearch(src_p, radius, PIdx, PDist);
		P_XYZ& dst_p = dstPC[i];
		if (PIdx.size() > 1)
		{
			int ptNum_ = PIdx.size();
			float inv_ = 1.0f / (float)PIdx.size();
			float sum_x = 0.0f, sum_y = 0.0f, sum_z = 0.0f;
			float sum_ = 0.0f;
			for (int j = 0; j < ptNum_; ++j)
			{
				P_XYZ& p_ = srcPC[PIdx[j]];
				sum_x += p_.x; sum_y += p_.y; sum_z += p_.z;
				sum_ += p_.x * p_.x + p_.y * p_.y + p_.z * p_.z;
			}
			sum_x *= inv_; sum_y *= inv_; sum_z *= inv_;
			float a = sum_ * inv_ - (sum_x * sum_x + sum_y * sum_y + sum_z * sum_z);

			a = a / (a + lamda);
			float b_x = sum_x - a * sum_x;
			float b_y = sum_y - a * sum_y;
			float b_z = sum_z - a * sum_z;
			dst_p = { a * src_p.x + b_x, a * src_p.y + b_y, a * src_p.z + b_z };
		}
		else
		{
			dst_p = src_p;
		}
	}
}
//===================================================================================

//降采样=============================================================================
void PC_DownSample(PC_XYZ& srcPC, PC_XYZ& dstPC, float size, int mode)
{
	size_t len = srcPC.points.size();
	if (len == 0)
		return;
	std::vector<bool> flags(len, false);
	pcl::KdTreeFLANN<P_XYZ> kdtree;
	kdtree.setInputCloud(srcPC.makeShared());
	for (size_t i = 0; i < len; ++i)
	{
		if (flags[i])
			continue;
		P_XYZ& ref_p = srcPC.points[i];
		std::vector<int> P_Idx;
		std::vector<float> P_Dist;
		if (mode == 0)
			kdtree.radiusSearch(ref_p, size, P_Idx, P_Dist);
		else
			kdtree.nearestKSearch(ref_p, size, P_Idx, P_Dist);
		float sum_x = 0.0f, sum_y = 0.0f, sum_z = 0.0f;
		for (int j = 0; j < P_Idx.size(); ++j)
		{
			P_XYZ& p_ = srcPC.points[P_Idx[j]];
			sum_x += p_.x; sum_y += p_.y; sum_z += p_.z;
			flags[P_Idx[j]] = true;
		}
		dstPC.points.push_back({ sum_x / P_Idx.size(), sum_y / P_Idx.size(), sum_z / P_Idx.size() });
	}
}
////===================================================================================

void PC_FitlerTest()
{
	string pc_rotpath = "D:/data/变形测试点云/EV/m_SrcPC_14.ply";
	PC_XYZ srcPC;
	pcl::io::loadPLYFile(pc_rotpath, srcPC);

	PC_XYZ samplePC;
	PC_DownSample(srcPC, samplePC, 1.5, 0);

	PC_XYZ dstPC;
	PC_GuideFilter(samplePC, dstPC, 20, 20.0);

	pcl::visualization::PCLVisualizer viewer;
	//显示轨迹
	pcl::visualization::PointCloudColorHandlerCustom<P_XYZ> red(samplePC.makeShared(), 255, 0, 0);
	viewer.addPointCloud(samplePC.makeShared(), red, "samplePC");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "samplePC");
	pcl::visualization::PointCloudColorHandlerCustom<P_XYZ> green(dstPC.makeShared(), 0, 255, 0);
	viewer.addPointCloud(dstPC.makeShared(), green, "dstPC");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "dstPC");

	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}
}