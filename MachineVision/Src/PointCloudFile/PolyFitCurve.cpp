#include "../../include/PointCloudFile/PolyFitCurve.h"

//获取参数T==========================================================================
void GetCurveTParam(vector<vector<float>>& t, int size, int order, float min_t, float max_t)
{
	float step = (max_t - min_t) / float(size);
	t.resize(size);
	for (int i = 0; i < size; ++i)
	{
		t[i].resize(order);
	}
	for (int i = 0; i < size; ++i)
	{
		float t_s = min_t + i * step;
		for (int k = 0; k < order; ++k)
		{
			t[i][k] = std::pow(t_s, k + 1);
		}
	}
}
//===================================================================================

//获取参数矩阵A======================================================================
void CalCurveMatA(Eigen::MatrixXf& matA, vector<vector<float>>& t)
{
	int order = t[0].size();
	int size = t.size();
	matA = Eigen::MatrixXf::Zero(order, order);
	for (int k1 = 0; k1 < order; ++k1)
	{
		for (int k2 = 0; k2 < k1 + 1; ++k2)
		{
			float& a_ele = matA(k1, k2);
			for (int i = 0; i < size; ++i)
			{
				a_ele += t[i][k1] * t[i][k2];
			}
		}
	}
	for (int k1 = 0; k1 < order; ++k1)
	{
		for (int k2 = k1 + 1; k2 < order; ++k2)
		{
			matA(k1, k2) = matA(k2, k1);
		}
	}
}
//===================================================================================

//获取参数矩阵B======================================================================
void CalCurveMatB(Eigen::MatrixXf& matB, vector<vector<float>>& t, vector<float>& x)
{
	int order = t[0].size();
	int size = t.size();
	matB = Eigen::MatrixXf::Zero(order, 1);
	for (int k = 0; k < order; ++k)
	{
		float& b_ele = matB(k, 0);
		for (int i = 0; i < size; ++i)
		{
			b_ele += t[i][k] * x[i];
		}
	}
}
//===================================================================================

//拟合多项式=========================================================================
void FitCurvePoly(vector<float>& x, vector<vector<float>>& t, vector<float>& coff, int order)
{
	int size = x.size();
	vector<float> sum_t(order, 0);
	for (int k = 0; k < order; ++k)
	{
		for (int i = 0; i < size; ++i)
		{
			sum_t[k] += t[i][k];
		}
		sum_t[k] /= float(size);
	}
	float sum_x = 0.0f;
	for (int i = 0; i < size; ++i)
	{
		for (int k = 0; k < order; ++k)
		{
			t[i][k] -= sum_t[k];
		}
		sum_x += x[i];
	}
	vector<float> x_(size);
	sum_x /= float(size);
	for (int i = 0; i < size; ++i)
	{
		x_[i] = x[i] - sum_x;
	}

	Eigen::MatrixXf matA;
	Eigen::MatrixXf matB;
	CalCurveMatA(matA, t);
	CalCurveMatB(matB, t, x_);
	Eigen::MatrixXf coff_ = matA.inverse() * matB;
	coff.resize(order + 1);

	float a0 = 0;
	for (int k = 0; k < order; ++k)
	{
		coff[k] = coff_(k, 0);
		a0 += coff[k] * sum_t[k];
	}
	coff[order] = sum_x - a0;
}
//===================================================================================

//多项式平滑=========================================================================
void PolyCurveSmooth(PC_XYZ& srcPC, PC_XYZ& dstPC, int size, int order, float min_t, float max_t)
{
	int ptsNum = srcPC.size();
	int halfSize = size / 2;

	vector<float> x(size);
	vector<float> y(size);
	vector<float> z(size);
	vector<vector<float>> t;
	GetCurveTParam(t, size, order, min_t, max_t);

	dstPC.resize(ptsNum);
	for (int i = halfSize; i < ptsNum - halfSize; ++i)
	{
		int idx = 0;
		for (int j = i - halfSize; j <= i + halfSize; ++j)
		{
			x[idx] = srcPC[j].x;
			y[idx] = srcPC[j].y;
			z[idx] = srcPC[j].z;
			++idx;
		}

		vector<float> coff_x, coff_y, coff_z;
		FitCurvePoly(x, t, coff_x, order);
		FitCurvePoly(y, t, coff_y, order);
		FitCurvePoly(z, t, coff_z, order);

		float x = 0, y = 0, z = 0;
		for (int k = 0; k < order; ++k)
		{
			x += t[halfSize][k] * coff_x[k];
			y += t[halfSize][k] * coff_y[k];
			z += t[halfSize][k] * coff_z[k];
		}
		x += coff_x[order];
		y += coff_y[order];
		z += coff_z[order];
		dstPC[i] = { x, y, z };
	}
}
//===================================================================================

void PolyCurveFitmoothTest()
{
	PC_XYZ srcTrack;
	pcl::io::loadPLYFile("D:/data/变形测试点云/EV/FitPolyPC.ply", srcTrack);

	PC_XYZ dstTrack1;
	int order = 2;

	PolyCurveSmooth(srcTrack, dstTrack1, 55, order, -3, 3);

	pcl::visualization::PCLVisualizer viewer;
	viewer.addCoordinateSystem(10);
	//显示轨迹
	pcl::visualization::PointCloudColorHandlerCustom<P_XYZ> red(srcTrack.makeShared(), 255, 0, 0); //设置点云颜色
	viewer.addPointCloud(srcTrack.makeShared(), red, "srcTrack");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "srcTrack");

	pcl::visualization::PointCloudColorHandlerCustom<P_XYZ> green(dstTrack1.makeShared(), 0, 255, 0); //设置点云颜色
	viewer.addPointCloud(dstTrack1.makeShared(), green, "dstTrack1");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "dstTrack1");

	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}
}