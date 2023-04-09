#include "../../include/FitShapeFile/PC_FitPlane.h"
#include "../../include/FitShapeFile/ComputeModels.h"
#include "../../include/BaseOprFile/MathOpr.h"

//���һ�²����㷨����ƽ��======================================================================
void PC_RANSACFitPlane(NB_Array3D pts, Plane3D& plane, vector<int>& inliners, double thres)
{
	if (pts.size() < 6)
		return;
	int best_model_p = 0;
	double P = 0.99;  //ģ�ʹ��ڵĸ���
	double log_P = log(1 - P);
	int size = pts.size();
	int maxEpo = 10000;
	vector<Point3d> pts_(3);
	for (int i = 0; i < maxEpo; ++i)
	{
		//��ֹ������ѭ��
		if (i > 500)
			break;
		int effetPoints = 0;
		//���ѡ�����������ƽ��---ע�⣺���������Ҫ���⴦���ֹ����ͬ
		pts_[0] = pts[rand() % size]; pts_[1] = pts[rand() % size];	pts_[2] = pts[rand() % size];
		Plane3D plane_;
		PC_ThreePtsComputePlane(pts_[0], pts_[1], pts_[2], plane_);
		//������ڵ�ĸ���
		for (int j = 0; j < size; ++j)
		{
			effetPoints += PC_PtToPlaneDist(pts[j], plane_) < thres ? 1 : 0;
		}
		//��ȡ����ģ�ͣ������ݸ����޸ĵ�������
		if (best_model_p < effetPoints)
		{
			best_model_p = effetPoints;
			plane = plane_;
			double t_P = (double)best_model_p / size;
			double pow_t_p = t_P * t_P * t_P;
			maxEpo = log_P / log(1 - pow_t_p) + std::sqrt(1 - pow_t_p) / (pow_t_p);
		}
	}
	//��ȡ���ڵ�
	if (inliners.size() != 0)
		inliners.resize(0);
	inliners.reserve(size);
	for (int i = 0; i < size; ++i)
	{
		if (PC_PtToPlaneDist(pts[i], plane) < thres)
			inliners.push_back(i);
	}
}
//==============================================================================================

//��С���˷����ƽ��============================================================================
void PC_OLSFitPlane(NB_Array3D pts, vector<double>& weights, Plane3D& plane)
{
	if (pts.size() < 3)
		return;
	double w_sum = 0.0;
	double w_x_sum = 0.0;
	double w_y_sum = 0.0;
	double w_z_sum = 0.0;
	for (int i = 0; i < pts.size(); ++i)
	{
		w_sum += weights[i];
		w_x_sum += weights[i] * pts[i].x;
		w_y_sum += weights[i] * pts[i].y;
		w_z_sum += weights[i] * pts[i].z;
	}
	w_sum = 1.0 / std::max(w_sum, EPS);
	double w_x_mean = w_x_sum * w_sum;
	double w_y_mean = w_y_sum * w_sum;
	double w_z_mean = w_z_sum * w_sum;

	cv::Mat A(3, 3, CV_64FC1, cv::Scalar(0));
	double *pA = A.ptr<double>(0);
	for (int i = 0; i < pts.size(); ++i)
	{
		double x_ = pts[i].x - w_x_mean;
		double y_ = pts[i].y - w_y_mean;
		double z_ = pts[i].z - w_z_mean;
		pA[0] += weights[i] * x_ * x_;
		pA[4] += weights[i] * y_ * y_;
		pA[8] += weights[i] * z_ * z_;
		pA[1] += weights[i] * x_ * y_;
		pA[2] += weights[i] * x_ * z_;
		pA[5] += weights[i] * y_ * z_;
	}
	pA[3] = pA[1]; pA[6] = pA[2]; pA[7] = pA[5];
	cv::Mat eigenVal, eigenVec;
	cv::eigen(A, eigenVal, eigenVec);
	double* pEigenVec = eigenVec.ptr<double>(2);
	plane.a = pEigenVec[0]; plane.b = pEigenVec[1]; plane.c = pEigenVec[2];
	plane.d = -(plane.a * w_x_mean + plane.b * w_y_mean + plane.c * w_z_mean);
}
//==============================================================================================

//Huber����Ȩ��=================================================================================
void PC_HuberPlaneWeights(NB_Array3D pts, Plane3D& plane, vector<double>& weights)
{
	double tao = 1.345;
	for (int i = 0; i < pts.size(); ++i)
	{
		double dist = PC_PtToPlaneDist(pts[i], plane);
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
//==============================================================================================

//Tukey����Ȩ��================================================================================
void PC_TukeyPlaneWeights(NB_Array3D pts, Plane3D& plane, vector<double>& weights)
{
	vector<double> dists(pts.size());
	for (int i = 0; i < pts.size(); ++i)
	{
		dists[i] = PC_PtToPlaneDist(pts[i], plane);
	}
	//����������tao
	vector<double> disttanceSort = dists;
	sort(disttanceSort.begin(), disttanceSort.end());
	double tao = disttanceSort[(disttanceSort.size() - 1) / 2] / 0.6745 * 2;

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
//==============================================================================================

//ƽ�����======================================================================================
//template <typename T1, typename T2>
void PC_FitPlane(NB_Array3D pts, Plane3D& plane, int k, NB_MODEL_FIT_METHOD method)
{
	vector<double> weights(pts.size(), 1);
	PC_OLSFitPlane(pts, weights, plane);
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
				PC_HuberPlaneWeights(pts, plane, weights);
				break;
			case TUKEY_FIT:
				PC_TukeyPlaneWeights(pts, plane, weights);
				break;
			default:
				break;
			}
			PC_OLSFitPlane(pts, weights, plane);
		}
	}
}
//==============================================================================================

//�ռ�ƽ����ϲ���==============================================================================
void PC_FitPlaneTest()
{
	PC_XYZ::Ptr srcPC(new PC_XYZ);
	pcl::io::loadPLYFile("F:/nbcode/image/testimage/����ƽ��.ply", *srcPC);

	vector<P_XYZ> pts(srcPC->points.size());
	for (int i = 0; i < srcPC->points.size(); ++i)
	{
		pts[i] = srcPC->points[i];
	}

	std::random_shuffle(pts.begin(), pts.end());
	Plane3D plane;
	vector<int> inliners;
	PC_FitPlane(pts, plane, 5, NB_MODEL_FIT_METHOD::TUKEY_FIT);

	PC_RANSACFitPlane(pts, plane, inliners, 0.01);

	PC_XYZ::Ptr inlinerPC(new PC_XYZ);
	inlinerPC->points.resize(inliners.size());
	for (int i = 0; i < inliners.size(); ++i)
	{
		inlinerPC->points[i] = pts[inliners[i]];
	}
	pcl::visualization::PCLVisualizer viewer;
	viewer.addCoordinateSystem(10);
	pcl::visualization::PointCloudColorHandlerCustom<P_XYZ> write(inlinerPC, 255, 255, 255); //���õ�����ɫ
	viewer.addPointCloud(inlinerPC, write, "inlinerPC");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "inlinerPC");

	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}
}
//============================================================================================
