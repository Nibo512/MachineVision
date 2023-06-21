#include "../../include/FitShapeFile/PC_FitCircle.h"
#include "../../include/FitShapeFile/PC_FitPlane.h"
#include "../../include/FitShapeFile/ComputeModels.h"
#include "../../include/BaseOprFile/MathOpr.h"
#include <boost/algorithm/string/split.hpp>

//随机一致采样算法计算园==========================================================================
void PC_RANSACFitCircle(NB_Array3D pts, Circle3D& circle, vector<int>& inliners, double thres)
{
	if (pts.size() < 3)
		return;
	int best_model_p = 0;
	double P = 0.99;  //模型存在的概率
	double log_P = log(1 - P);
	int size = pts.size();
	int maxEpo = 10000;
	for (int i = 0; i < maxEpo; ++i)
	{
		int effetPoints = 0;
		//随机选择三个点计算园---注意：这里可能需要特殊处理防止点相同
		int index_1 = rand() % size; int index_2 = rand() % size; int index_3 = rand() % size;
		Circle3D circle_;
		PC_ThreePtsComputeCircle(pts[index_1], pts[index_2], pts[index_3], circle_);
		//计算局内点的个数
		for (int j = 0; j < size; ++j)
		{
			effetPoints += PC_PtToCircleDist(pts[j], circle_) < thres ? 1 : 0;
		}
		//获取最优模型，并根据概率修改迭代次数
		if (best_model_p < effetPoints)
		{
			best_model_p = effetPoints;
			circle = circle_;
			double t_P = (double)best_model_p / size;
			double pow_t_p = t_P * t_P * t_P;
			maxEpo = log_P / log(1 - pow_t_p) + std::sqrt(1 - pow_t_p) / (pow_t_p);
		}
	}
	//提取局内点
	if (inliners.size() != 0)
		inliners.resize(0);
	inliners.reserve(size);
	for (int i = 0; i < size; ++i)
	{
		if (PC_PtToCircleDist(pts[i], circle) < thres)
			inliners.push_back(i);
	}
}
//================================================================================================

//最小二乘法拟合空间空间园========================================================================
void PC_OLSFit3DCircle(NB_Array3D pts, vector<double>& weights, Circle3D& circle, Plane3D& plane)
{
	double w_sum = 0.0;
	double w_x_sum = 0.0;
	double w_y_sum = 0.0;
	double w_z_sum = 0.0;
	double w_x2y2z2_sum = 0.0;
	for (int i = 0; i < pts.size(); ++i)
	{
		w_sum += weights[i];
		w_x_sum += weights[i] * pts[i].x;
		w_y_sum += weights[i] * pts[i].y;
		w_z_sum += weights[i] * pts[i].z;
		w_x2y2z2_sum += weights[i] * (pts[i].x * pts[i].x + pts[i].y * pts[i].y + pts[i].z * pts[i].z);
	}
	w_sum = 1.0 / std::max(w_sum, EPS);
	double w_x_mean = w_x_sum * w_sum;
	double w_y_mean = w_y_sum * w_sum;
	double w_z_mean = w_z_sum * w_sum;
	double w_x2y2z2_mean = w_x2y2z2_sum * w_sum;

	double a = plane.a, b = plane.b, c = plane.c;
	Mat A(3, 3, CV_64FC1, cv::Scalar(0));
	Mat B(3, 1, CV_64FC1, cv::Scalar(0));
	double* pA = A.ptr<double>(0);
	double* pB = B.ptr<double>(0);
	for (int i = 0; i < pts.size(); ++i)
	{
		double x = pts[i].x, y = pts[i].y, z = pts[i].z, w = weights[i];
		double x_ = x - w_x_mean, y_ = y - w_y_mean, z_ = z - w_z_mean;
		pA[0] += w * (x_ * x_ + 0.25 * a * a);
		pA[1] += w * (x_ * y_ + 0.25 * a * b);
		pA[2] += w * (x_ * z_ + 0.25 * c * a);
		pA[4] += w * (y_ * y_ + 0.25 * b * b);
		pA[5] += w * (y_ * z_ + 0.25 * b * c);
		pA[8] += w * (z_ * z_ + 0.25 * c * c);

		double r_ = pts[i].x * pts[i].x + pts[i].y * pts[i].y + pts[i].z * pts[i].z - w_x2y2z2_mean;
		pB[0] -= w * (x_ * r_ + 0.5 * a * a * x + 0.5 * a * b * y + 0.5 * a * c * z);
		pB[1] -= w * (y_ * r_ + 0.5 * a * b * x + 0.5 * b * b * y + 0.5 * b * c * z);
		pB[2] -= w * (z_ * r_ + 0.5 * a * c * x + 0.5 * b * c * y + 0.5 * c * c * z);
	}
	pA[3] = pA[1]; pA[6] = pA[2]; pA[7] = pA[5];

	Mat C = (A.inv()) * B;
	double* pC = C.ptr<double>(0);
	circle.x = -pC[0] / 2.0;
	circle.y = -pC[1] / 2.0;
	circle.z = -pC[2] / 2.0;
	double c_ = -(pC[0] * w_x_mean + pC[1] * w_y_mean + pC[2] * w_z_mean + w_x2y2z2_mean);
	circle.r = std::sqrt(std::max(circle.x * circle.x + circle.y * circle.y + circle.z * circle.z - c_, EPS));
	circle.a = a; circle.b = b; circle.c = c;
}
//================================================================================================

//Huber计算权重===================================================================================
void PC_HuberCircleWeights(NB_Array3D pts, Circle3D& circle, vector<double>& weights)
{
	double tao = 1.345;
	for (int i = 0; i < pts.size(); ++i)
	{
		double dist = PC_PtToCircleDist(pts[i], circle);
		if (dist <= tao)
		{
			weights[i] = 1;
		}
		else
		{
			weights[i] = tao / (dist);
		}
	}
}
//================================================================================================

//Tukey计算权重===================================================================================
void PC_TukeyCircleWeights(NB_Array3D pts, Circle3D& circle, vector<double>& weights)
{
	vector<double> dists(pts.size());
	for (int i = 0; i < pts.size(); ++i)
	{
		dists[i] = PC_PtToCircleDist(pts[i], circle);
	}
	//求限制条件tao
	vector<double> disttanceSort = dists;
	sort(disttanceSort.begin(), disttanceSort.end());
	double tao = disttanceSort[(disttanceSort.size() - 1) / 2] / 0.6745 * 2;

	//更新权重
	for (int i = 0; i < dists.size(); ++i)
	{
		if (dists[i] <= tao)
		{
			double d_tao = dists[i] / tao;
			weights[i] = std::pow((1 - d_tao * d_tao), 2);
		}
		else weights[i] = 0;
	}
}
//================================================================================================

//Tukey计算权重===================================================================================
void PC_TukeyCircleWeights_(NB_Array3D pts, Circle3D& circle, vector<double>& weights)
{
	vector<double> dists(pts.size());
	for (int i = 0; i < pts.size(); ++i)
	{
		dists[i] = PC_PtToCircleDist(pts[i], circle);
	}
	//求限制条件tao
	vector<double> disttanceSort = dists;
	sort(disttanceSort.begin(), disttanceSort.end());
	double tao = disttanceSort[(disttanceSort.size() - 1) / 2] / 0.6745 * 2;

	//更新权重
	for (int i = 0; i < dists.size(); ++i)
	{
		if (dists[i] <= tao)
		{
			double d_tao = dists[i] / tao;
			weights[i] = std::pow((1 - d_tao * d_tao), 2);
		}
		else
		{
			double d_tao =  2 * dists[i] / tao ;
			weights[i] = std::exp(-d_tao);
		}
	}
}
//================================================================================================

//拟合圆==========================================================================================
void PC_FitCircle(NB_Array3D pts, Circle3D& circle, int k, NB_MODEL_FIT_METHOD method)
{
	//先拟合平面
	Plane3D plane;
	PC_FitPlane(pts, plane, k, method);

	vector<double> weights(pts.size(), 1);
	PC_TukeyPlaneWeights(pts, plane, weights);

	PC_OLSFit3DCircle(pts, weights, circle, plane);
	if (method == NB_MODEL_FIT_METHOD::OLS_FIT)
	{
		return;
	}
	else
	{
		for (int i = 0; i < k; ++i)
		{
			switch (method)
			{
			case HUBER_FIT:
				PC_HuberCircleWeights(pts, circle, weights);
				break;
			case TUKEY_FIT:
				PC_TukeyCircleWeights(pts, circle, weights);
				break;
			case TUKEY_FITIMP:
				PC_TukeyCircleWeights_(pts, circle, weights);
				break;
			default:
				break;
			}
			PC_OLSFit3DCircle(pts, weights, circle, plane);
		}
	}
}
//================================================================================================

//计算平整度======================================================================================
void ComputeCircleFlatness(PC_XYZ::Ptr srcPC, Circle3D& circle, int idx)
{
	fstream res("D:/res" + to_string(idx) + ".csv", ios::app);
	//res << "a" << "," << "b" << "," << "c" << "," << "x" << "," << "y" << "," << "z" << "," << "r" << endl;
	//res << circle.a << "," << circle.b << "," << circle.c << "," << circle.x << "," << circle.y << "," << circle.z << "," << circle.r << endl;
	res << "平整度" << "," << "圆度" << endl;
	double norm_ = std::sqrt(circle.a * circle.a + circle.b * circle.b + circle.c * circle.c);
	float d = -(circle.x * circle.a + circle.y * circle.b + circle.z * circle.c);
	for (int i = 0; i < srcPC->size(); ++i)
	{
		P_XYZ& pt = srcPC->points[i];
		//计算平整度
		double flati = (pt.x * circle.a + pt.y * circle.b + pt.z * circle.c + d) / norm_;

		//计算圆度
		double diff_x = pt.x - circle.x;
		double diff_y = pt.y - circle.y;
		double diff_z = pt.z - circle.z;
		double ri = std::sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z);
		double roundnessi = ri - circle.r;

		res << flati * 1000<< "," << roundnessi * 1000 << endl;
	}

	res.close();
}
//================================================================================================


//空间三维圆拟合测试==========================================================================
void PC_FitCircleTest()
{
	PC_XYZ::Ptr srcPC(new PC_XYZ), srcPC_1(new PC_XYZ);
	pcl::io::loadPLYFile("D:/data/点云数据/形状数据/实际圆5.ply", *srcPC);

	pcl::io::loadPLYFile("D:/data/点云数据/形状数据/实际圆5.ply", *srcPC_1);

	vector<P_XYZ> pts(srcPC->points.size());
	for (int i = 0; i < srcPC->points.size(); ++i)
	{
		pts[i] = srcPC->points[i];
	}
	Circle3D circle;
	circle.a = -0.3875; circle.b = -0.9226; circle.c = -0.0063;
	circle.x = 1008.1228; circle.y = 1018.8738; circle.z = 14.7159;
	circle.r = 3.0661;
	cout << circle.a << "," << circle.b << "," << circle.c << "," << circle.x << "," << circle.y << "," << circle.z << "," << circle.r << endl;
	//for (int i = 0; i < 1; ++i)
	//{
	//	vector<int> inliners;
	//	PC_RANSACFitCircle(pts, circle, inliners, 0.001);
	//	ComputeCircleFlatness(srcPC_1, circle, i + 1);
	//}

	//vector<P_XYZ> inlinerPC;
	//inlinerPC.resize(inliners.size());
	//for (int i = 0; i < inliners.size(); ++i)
	//{
	//	inlinerPC[i] = pts[inliners[i]];
	//}

	//PC_FitCircle(pts, circle, 10, NB_MODEL_FIT_METHOD::TUKEY_FIT);
	//cout << circle.a << "," << circle.b << "," << circle.c << "," << circle.x << "," << circle.y << "," << circle.z << "," << circle.r << endl;
	ComputeCircleFlatness(srcPC_1, circle);


	pcl::visualization::PCLVisualizer viewer;
	//viewer.addCoordinateSystem(1000);
	//显示轨迹
	pcl::visualization::PointCloudColorHandlerCustom<P_XYZ> red(srcPC_1, 255, 0, 0); //设置点云颜色
	viewer.addPointCloud(srcPC_1, red, "srcPC_1");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "srcPC_1");

	//pcl::visualization::PointCloudColorHandlerCustom<P_XYZ> write(inlinerPC, 255, 255, 255); //设置点云颜色
	//viewer.addPointCloud(inlinerPC, write, "spherePC");
	//viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "spherePC");
	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}
}
//============================================================================================
