#include "../../include/FitShapeFile/Img_FitLine.h"
#include "../../include/FitShapeFile/ComputeModels.h"
#include "../../include/MatchFile/ContourOpr.h"

//随机一致采样算法计算直线======================================================================
void Img_RANSACFitLine(NB_Array2D pts, Line2D& line, vector<int>& inliners, double thres)
{
	if (pts.size() < 2)
		return;
	int best_model_p = 0;
	double P = 0.99;  //模型存在的概率
	double log_P = log(1 - P);
	int size = pts.size();
	int maxEpo = 10000;
	for (int i = 0; i < maxEpo; ++i)
	{
		int effetPoints = 0;
		//随机选择两个点计算直线---注意：这里可能需要特殊处理防止点相同
		int index_1 = rand() % size;
		int index_2 = rand() % size;
		Line2D line_;
		Img_TwoPtsComputeLine(pts[index_1], pts[index_2], line_);

		//计算局内点的个数
		for (int j = 0; j < size; ++j)
		{
			effetPoints += abs(line_.a * pts[j].x + line_.b * pts[j].y + line_.c) < thres ? 1 : 0;
		}
		//获取最优模型，并根据概率修改迭代次数
		if (best_model_p < effetPoints)
		{
			best_model_p = effetPoints;
			line = line_;
			double t_P = (double)best_model_p / size;
			double pow_t_p = t_P * t_P;
			maxEpo = log_P / log(1 - pow_t_p) + std::sqrt(1 - pow_t_p) / (pow_t_p);
		}
	}

	//提取局内点
	if (inliners.size() != 0)
		inliners.resize(0);
	inliners.reserve(size);
	for (int i = 0; i < size; ++i)
	{
		if (abs(line.a * pts[i].x + line.b * pts[i].y + line.c) < thres)
			inliners.push_back(i);
	}
}
//==============================================================================================

//最小二乘法拟合直线============================================================================
void Img_OLSFitLine(NB_Array2D pts, vector<double>& weights, Line2D& line)
{
	double w_sum = 0.0;
	double w_x_sum = 0.0;
	double w_y_sum = 0.0;
	for (int i = 0; i < weights.size(); ++i)
	{
		w_sum += weights[i];
		w_x_sum += weights[i] * pts[i].x;
		w_y_sum += weights[i] * pts[i].y;
	}
	w_sum = 1.0 / std::max(w_sum, EPS);
	double w_x_mean = w_x_sum * w_sum;
	double w_y_mean = w_y_sum * w_sum;
	Mat A(2, 2, CV_64FC1, cv::Scalar(0));
	double* pA = A.ptr<double>(0);
	for (int i = 0; i < pts.size(); ++i)
	{
		double x_ = pts[i].x - w_x_mean;
		double y_ = pts[i].y - w_y_mean;
		pA[0] += weights[i] * x_ * x_;
		pA[1] += weights[i] * x_ * y_;
		pA[3] += weights[i] * y_ * y_;
	}
	pA[2] = pA[1];
	Mat eigenVal, eigenVec;
	eigenNonSymmetric(A, eigenVal, eigenVec);
	double* pEigenVec = eigenVec.ptr<double>(1);
	line.a = pEigenVec[0];
	line.b = pEigenVec[1];
	line.c = -(w_x_mean * line.a + w_y_mean * line.b);
}
//==============================================================================================

//Huber计算权重=================================================================================
void Img_HuberLineWeights(NB_Array2D pts, Line2D& line, vector<double>& weights)
{
	double tao = 1.345;
	for (int i = 0; i < pts.size(); ++i)
	{
		double distance = abs(pts[i].x * line.a + pts[i].y * line.b + line.c);
		if (distance <= tao)
		{
			weights[i] = 1;
		}
		else
		{
			weights[i] = tao / distance;
		}
	}
}
//==============================================================================================

//Tukey计算权重================================================================================
void Img_TukeyLineWeights(NB_Array2D pts, Line2D& line, vector<double>& weights)
{
	vector<double> dists(pts.size());
	for (int i = 0; i < pts.size(); ++i)
	{
		double distance = abs(pts[i].x * line.a + pts[i].y * line.b + line.c);
		dists[i] = distance;
	}
	vector<double> disttanceSort = dists;
	sort(disttanceSort.begin(), disttanceSort.end());
	double tao = disttanceSort[(disttanceSort.size() - 1) / 2] / 0.6745 * 2;

	tao = std::max(tao, 1e-12);
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

//直线拟合======================================================================================
void Img_FitLine(NB_Array2D pts, Line2D& line, int k, NB_MODEL_FIT_METHOD method)
{
	vector<double> weights(pts.size(), 1);
	Img_OLSFitLine(pts, weights, line);
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
				Img_HuberLineWeights(pts, line, weights);
				break;
			case TUKEY_FIT:
				Img_TukeyLineWeights(pts, line, weights);
				break;
			default:
				break;
			}
			Img_OLSFitLine(pts, weights, line);
		}
	}
}
//==============================================================================================

//二维直线拟合测试============================================================================
void Img_FitLineTest()
{
	//string imgPath = "D:/data/TestImage/直线拟合.bmp";
	//cv::Mat srcImg = cv::imread(imgPath, 0);
	//cv::Mat binImg;
	//cv::threshold(srcImg, binImg, 10, 255, ThresholdTypes::THRESH_BINARY_INV);
	//vector<vector<cv::Point>> contours;
	//cv::findContours(binImg, contours, RetrievalModes::RETR_LIST, ContourApproximationModes::CHAIN_APPROX_NONE);

	//vector<cv::Point2f> pts(contours.size());
	//for (int i = 0; i < contours.size(); ++i)
	//{
	//	int len = contours[i].size();
	//	float sum_x = 0.0f, sum_y = 0.0f;
	//	for (int j = 0; j < len; ++j)
	//	{
	//		sum_x += contours[i][j].x;
	//		sum_y += contours[i][j].y;
	//	}
	//	pts[i].x = sum_x / len;
	//	pts[i].y = sum_y / len;
	//}

	vector<cv::Point2f> pts(6);
	pts[0] = cv::Point2f(382, 120); pts[1] = cv::Point2f(403, 118);
	pts[2] = cv::Point2f(425, 118); pts[3] = cv::Point2f(447, 118);
	pts[4] = cv::Point2f(469, 118); pts[5] = cv::Point2f(490, 118);

	Line2D line;
	vector<int> inliners;
	//Img_RANSACFitLine(pts, line, inliners, 2);
	Img_FitLine(pts, line, 5, NB_MODEL_FIT_METHOD::TUKEY_FIT);

	cv::Mat image(cv::Size(800, 200), CV_8UC3, cv::Scalar(0));
	for (int i = 0; i < pts.size(); ++i)
	{
		cv::line(image, pts[i], pts[i], cv::Scalar(255,255,255), 3);
	}

	cv::Point s_pt, e_pt;
	s_pt.x = 360; s_pt.y = -(line.c + s_pt.x * line.a) / line.b;
	e_pt.x = 500; e_pt.y = -(line.c + e_pt.x * line.a) / line.b;

	//Mat colorImg;
	//cv::cvtColor(srcImg, colorImg, cv::COLOR_GRAY2BGR);
	//cv::line(colorImg, s_pt, e_pt, cv::Scalar(0, 255, 0), 1);


	//Img_FitLine(pts, line, 5, NB_MODEL_FIT_METHOD::TUKEY_FIT);
	//s_pt.x = 35; s_pt.y = -(line.c + 35 * line.a) / line.b;
	//e_pt.x = 800; e_pt.y = -(line.c + 800 * line.a) / line.b;
	cv::line(image, s_pt, e_pt, cv::Scalar(0, 0, 255), 1);
	//cv::imwrite("D:/data/TestImage/直线拟合Res.bmp", colorImg);

	//int width = colorImg.cols;
	//int height = colorImg.rows;
	//VideoWriter video;
	//video.open("D:/data/TestImage/直线拟合Res.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(width, height), true);
	//if (!video.isOpened())
	//	cout << "mp4创建失败!" << endl;

	//for (int i = 0; i < 100; ++i)
	//{
	//	for (int j = 1; j <= 6; ++j)
	//	{
	//		video.write(colorImg);
	//	}
	//}

	//for (int i = 0; i < inliners.size(); ++i)
	//{
	//	cv::line(colorImg, pts[inliners[i]], pts[inliners[i]], cv::Scalar(0, 0, 255), 1);
	//}
}
//============================================================================================