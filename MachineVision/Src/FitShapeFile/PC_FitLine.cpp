#include "../../include/FitShapeFile/PC_FitLine.h"
#include "../../include/FitShapeFile/ComputeModels.h"
#include "../../include/BaseOprFile/MathOpr.h"

//���һ�²����㷨����ռ�ֱ��====================================================================
void PC_RANSACFitLine(NB_Array3D pts, Line3D& line, vector<int>& inliners, double thres)
{
	if (pts.size() < 6)
		return;
	int best_model_p = 0;
	double P = 0.99;  //ģ�ʹ��ڵĸ���
	double log_P = log(1 - P);
	int size = pts.size();
	int maxEpo = 10000;
	for (int i = 0; i < maxEpo; ++i)
	{
		int effetPoints = 0;
		//���ѡ�����������ֱ��---ע�⣺���������Ҫ���⴦���ֹ����ͬ
		Point3d pt1 = pts[rand() % size]; 
		Point3d pt2 = pts[rand() % size];
		Line3D line_;
		PC_TwoPtsComputeLine(pt1, pt2, line_);
		//������ڵ�ĸ���
		for (int j = 0; j < size; ++j)
		{
			effetPoints += PC_PtToLineDist(pts[j], line_) < thres ? 1 : 0;
		}
		//��ȡ����ģ�ͣ������ݸ����޸ĵ�������
		if (best_model_p < effetPoints)
		{
			best_model_p = effetPoints;
			line = line_;
			double t_P = (double)best_model_p / size;
			double pow_t_p = t_P * t_P;
			maxEpo = log_P / log(1 - pow_t_p) + std::sqrt(1 - pow_t_p) / (pow_t_p);
		}
	}
	//��ȡ���ڵ�
	if (inliners.size() != 0)
		inliners.resize(0);
	inliners.reserve(size);
	for (int i = 0; i < size; ++i)
	{
		if (PC_PtToLineDist(pts[i], line) < thres)
			inliners.push_back(i);
	}
}
//================================================================================================

//��С���˷���Ͽռ�ֱ��==========================================================================
void PC_OLSFit3DLine(NB_Array3D pts, vector<double>& weights, Line3D& line)
{
	double w_sum = 0.0, w_x_sum = 0.0, w_y_sum = 0.0, w_z_sum = 0.0;
	double w_xy_sum = 0.0, w_yz_sum = 0.0, w_zx_sum = 0.0;
	for (int i = 0; i < pts.size(); ++i)
	{
		double w = weights[i], x = pts[i].x, y = pts[i].y, z = pts[i].z;
		w_x_sum += w * x; w_y_sum += w * y; w_z_sum += w * z; w_sum += w;
	}
	w_sum = 1.0 / std::max(w_sum, EPS);
	double w_x_mean = w_x_sum * w_sum;
	double w_y_mean = w_y_sum * w_sum;
	double w_z_mean = w_z_sum * w_sum;

	Mat A(3, 3, CV_64FC1, cv::Scalar(0));
	double* pA = A.ptr<double>(0);
	for (int i = 0; i < pts.size(); ++i)
	{
		double w = weights[i], x = pts[i].x, y = pts[i].y, z = pts[i].z;
		double x_ = x - w_x_mean;
		double y_ = y - w_y_mean;
		double z_ = z - w_z_mean;

		pA[0] += w * (y_ * y_ + z_ * z_);
		pA[1] -= w * x_ * y_;
		pA[2] -= w * z_ * x_;
		pA[4] += w * (x_ * x_ + z_ * z_);
		pA[5] -= w * y_ * z_;
		pA[8] += w * (x_ * x_ + y_ * y_);
	}
	pA[3] = pA[1]; pA[6] = pA[2]; pA[7] = pA[5];

	cv::Mat eigenVal, eigenVec;
	cv::eigen(A, eigenVal, eigenVec);
	double* pEigenVec = eigenVec.ptr<double>(2);
	line.a = pEigenVec[0]; line.b = pEigenVec[1]; line.c = pEigenVec[2];
	line.x = w_x_mean; line.y = w_y_mean; line.z = w_z_mean;
}
//================================================================================================

//Huber����Ȩ��===================================================================================
void PC_Huber3DLineWeights(NB_Array3D pts, Line3D& line, vector<double>& weights)
{
	double tao = 1.345;
	for (int i = 0; i < pts.size(); ++i)
	{
		double dist = PC_PtToLineDist(pts[i], line);
		if (dist <= tao)
		{
			weights[i] = 1;
		}
		else
		{
			weights[i] = tao / dist;
		}
	}
}
//================================================================================================

//Tukey����Ȩ��==================================================================================
void PC_Tukey3DLineWeights(NB_Array3D pts, Line3D& line, vector<double>& weights)
{
	vector<double> dists(pts.size(), 0.0);
	for (int i = 0; i < pts.size(); ++i)
	{
		dists[i] = PC_PtToLineDist(pts[i], line);
	}
	//����������tao
	vector<double> disttanceSort = dists;
	sort(disttanceSort.begin(), disttanceSort.end());
	double tao = disttanceSort[(disttanceSort.size() - 1) / 2] / 0.6745 * 2;

	tao = std::max(tao, 1e-12);
	//����Ȩ��
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

//�ռ�ֱ�����====================================================================================
void PC_Fit3DLine(NB_Array3D pts, Line3D& line, int k, NB_MODEL_FIT_METHOD method)
{
	vector<double> weights(pts.size(), 1);
	PC_OLSFit3DLine(pts, weights, line);
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
				PC_Huber3DLineWeights(pts, line, weights);
				break;
			case TUKEY_FIT:
				PC_Tukey3DLineWeights(pts, line, weights);
				break;
			default:
				break;
			}
			PC_OLSFit3DLine(pts, weights, line);
		}
	}
}
//==============================================================================================

//�ռ���λֱ����ϲ���========================================================================
void PC_FitLineTest()
{
	PC_XYZ::Ptr srcPC(new PC_XYZ);
	pcl::io::loadPLYFile("C:/Users/Administrator/Desktop/testimage/����ֱ��.ply", *srcPC);

	vector<P_XYZ> pts(srcPC->points.size());
	for (int i = 0; i < srcPC->points.size(); ++i)
	{
		pts[i] = srcPC->points[i];
	}
	std::random_shuffle(pts.begin(), pts.end());

	Line3D line;
	//PC_Fit3DLine(pts, line, 5, NB_MODEL_FIT_METHOD::TUKEY_FIT);

	vector<int> inliners;
	PC_RANSACFitLine(pts, line, inliners, 0.2);

	PC_XYZ::Ptr inlinerPC(new PC_XYZ);
	inlinerPC->points.resize(inliners.size());
	for (int i = 0; i < inliners.size(); ++i)
	{
		inlinerPC->points[i] = pts[inliners[i]];
	}

	pcl::visualization::PCLVisualizer viewer;
	viewer.addCoordinateSystem(10);
	//��ʾ�켣
	pcl::visualization::PointCloudColorHandlerCustom<P_XYZ> red(srcPC, 255, 0, 0); //���õ�����ɫ
	viewer.addPointCloud(srcPC, red, "srcPC");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "srcPC");

	pcl::visualization::PointCloudColorHandlerCustom<P_XYZ> write(inlinerPC, 255, 255, 255); //���õ�����ɫ
	viewer.addPointCloud(inlinerPC, write, "inlinerPC");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "inlinerPC");
	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}
}
//============================================================================================
