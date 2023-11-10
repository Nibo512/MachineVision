#include "../../include/PointCloudFile/PC_Filter.h"
#include "../../include/PointCloudFile/PointCloudOpr.h"
#include "../../include/PointCloudFile/PC_Seg.h"
#include <queue>

//随机采样一致性的点云分割==========================================================
void PC_RANSACSeg(PC_XYZ &srcPC, PC_XYZ &dstPC, int mode, float thresVal)
{
	if (mode > 16)
		return;
	if (srcPC.empty())
		return;

	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

	pcl::SACSegmentation<P_XYZ> seg;

	pcl::PointIndices::Ptr inliers(new pcl::PointIndices());

	seg.setOptimizeCoefficients(true);
	seg.setModelType(mode);
	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setMaxIterations(1000);
	seg.setDistanceThreshold(thresVal);
	seg.setInputCloud(srcPC.makeShared());
	seg.segment(*inliers, *coefficients);

	pcl::ExtractIndices<P_XYZ> extract;
	extract.setInputCloud(srcPC.makeShared());
	extract.setIndices(inliers);
	extract.setNegative(false);
	extract.filter(dstPC);
}
//==================================================================================

//基于欧式距离分割方法==============================================================
void PC_EuclideanSeg(PC_XYZ &srcPC, std::vector<P_IDX> &clusters, float distThresVal)
{
	if (srcPC.empty())
		return;
	pcl::search::KdTree<P_XYZ>::Ptr kdtree(new pcl::search::KdTree<P_XYZ>);
	kdtree->setInputCloud(srcPC.makeShared());
	pcl::EuclideanClusterExtraction<P_XYZ> clustering;
	clustering.setClusterTolerance(distThresVal);
	clustering.setMinClusterSize(1);
	clustering.setMaxClusterSize(10000000);
	clustering.setSearchMethod(kdtree);
	clustering.setInputCloud(srcPC.makeShared());
	clustering.extract(clusters);
}
//==================================================================================

//DBSCAN分割==========================================================================
void PC_DBSCANSeg(PC_XYZ &srcPC, vector<vector<int>> &indexs, double radius, int n, int minGroup, int maxGroup)
{
	int length = srcPC.size();

	//-1表示该点已经聚类、0表示该点为被聚类且不为核心点、1表示该点为核心点---只有核心点才能成为种子点
	vector<int> isLabeled(length, 0);
	pcl::KdTreeFLANN<P_XYZ> kdtree;
	kdtree.setInputCloud(srcPC.makeShared());
	//将点云中的点分为核心与非核心
#pragma omp parallel for
	for (int i = 0; i < length; ++i)
	{
		vector<int> PIdx;
		vector<float> DistIdx;
		kdtree.radiusSearch(srcPC[i], radius, PIdx, DistIdx);
		if (PIdx.size() > n)
		{
			isLabeled[i] = 1;
		}
	}

	//聚类
	queue<int> sands;
	for (int i = 0; i < isLabeled.size(); ++i)
	{
		if (isLabeled[i] == 1)
		{
			sands.push(i);
		}
		else
			continue;
		vector<int> index(0);
		while (!sands.empty())
		{
			vector<int> PIdx;
			vector<float> DistIdx;
			kdtree.radiusSearch(srcPC[sands.front()], radius, PIdx, DistIdx);
			for (int i = 0; i < PIdx.size(); ++i)
			{
				int idx = PIdx[i];
				if (isLabeled[idx] > -1)
				{
					index.push_back(idx);
					if (isLabeled[idx] == 1)
					{
						sands.push(idx);
					}
					isLabeled[idx] = -1;
				}
			}
			sands.pop();
		}
		if (index.size() > minGroup && index.size() < maxGroup)
			indexs.push_back(index);
	}
}
//===================================================================================

//Different Of Normal分割============================================================
void PC_DONSeg(PC_XYZ &srcPC, vector<int> &indexs, double large_r, double small_r, double thresVal)
{
	size_t length = srcPC.size();
	pcl::search::Search<P_XYZ>::Ptr tree;
	pcl::NormalEstimation<P_XYZ, P_N> ne;
	ne.setInputCloud(srcPC.makeShared());
	ne.setSearchMethod(tree);
	ne.setRadiusSearch(small_r);
	PC_N::Ptr normals_small_scale(new PC_N);
	ne.compute(*normals_small_scale);

	PC_N::Ptr normals_large_scale(new PC_N);
	ne.setRadiusSearch(large_r);
	ne.compute(*normals_large_scale);

	indexs.reserve(length);
	for (size_t i = 0; i < length; ++i)
	{
		P_N l_pn = normals_large_scale->points[i];
		P_N s_pn = normals_small_scale->points[i];
		float diff_x = std::fabs(abs(l_pn.normal_x) - abs(s_pn.normal_x)) * 0.5;
		float diff_y = std::fabs(abs(l_pn.normal_y) - abs(s_pn.normal_y)) * 0.5;
		float diff_z = std::fabs(abs(l_pn.normal_z) - abs(s_pn.normal_z)) * 0.5;
		if (diff_x > thresVal || diff_y > thresVal || diff_z > thresVal)
		{
			indexs.push_back(i);
		}
	}
}
//===================================================================================

//根据平面分割=======================================================================
void PC_SegBaseOnPlane(PC_XYZ &srcPC, Plane3D &plane, vector<int> &index, double thresVal, int orit)
{
	index.reserve(srcPC.size());
	for (int i = 0; i < srcPC.size(); ++i)
	{
		P_XYZ& p = srcPC[i];
		float dist = p.x * plane.a + p.y * plane.b + p.z * plane.c + plane.d;
		if (dist < thresVal && orit == 0)
		{
			index.push_back(i);
		}
		if (dist > thresVal && orit == 1)
		{
			index.push_back(i);
		}
	}
}
//===================================================================================

//根据曲率分割点云分割===============================================================
void PC_CurvatureSeg(PC_XYZ &srcPC, PC_N &normals, vector<int> &indexs, double H_Thres, double L_Thres)
{
	int pts_nun = srcPC.size();
	indexs.reserve(pts_nun);
	for (int i = 0; i < pts_nun; ++i)
	{
		double curvature = normals[i].curvature;
		if (curvature < L_Thres || curvature > H_Thres)
		{
			indexs.push_back(i);
		}
	}
}
//===================================================================================

/*点云分割测试程序*/
void PC_SegTest()
{
	PC_XYZ srcPC;
	string path = "D:/samplePC.ply";
	pcl::io::loadPLYFile(path, srcPC);
	//PC_XYZ downSrcPC;
	//PC_VoxelGrid(srcPC, downSrcPC, 0.2);
	vector<int> indexs;

	pcl::search::Search<P_XYZ>::Ptr tree;
	pcl::NormalEstimation<P_XYZ, P_N> ne;
	ne.setInputCloud(srcPC.makeShared());
	ne.setSearchMethod(tree);
	ne.setRadiusSearch(5.0);
	PC_N normals;
	ne.compute(normals);

	PC_CurvatureSeg(srcPC, normals, indexs, 100000, 0.001);
	PC_XYZ dstPC;
	PC_ExtractPC(srcPC, indexs, dstPC);

	std::vector<P_IDX> clusters;
	PC_EuclideanSeg(dstPC, clusters, 3.0);

	for (int i = 0; i < clusters.size(); ++i)
	{
		PC_XYZ selPC;
		PC_ExtractPC(dstPC, clusters[1].indices, selPC);

		pcl::visualization::PCLVisualizer viewer("srcPC");
		viewer.addCoordinateSystem(10);
		//显示轨迹
		pcl::visualization::PointCloudColorHandlerCustom<P_XYZ> red(srcPC.makeShared(), 255, 0, 0); //设置点云颜色
		viewer.addPointCloud(srcPC.makeShared(), red, "srcPC");
		viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "srcPC");

		pcl::visualization::PointCloudColorHandlerCustom<P_XYZ> green(selPC.makeShared(), 0, 255, 0); //设置点云颜色
		viewer.addPointCloud(selPC.makeShared(), green, "selPC");
		viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 8, "selPC");

		while (!viewer.wasStopped())
		{
			viewer.spinOnce();
		}
	}
}