#include <omp.h>
#include "../../include/MatchFile/ContourOpr.h"
#include "../../include/MatchFile/ShapeMatch.h"

//����ģ��=============================================================================
bool CreateShapeModel(Mat &modImg, ShapeModel* &model, SPAPLEMODELINFO &shapeModelInfo)
{
	ClearModel(model);
	model = new ShapeModel;

	model->angStep = shapeModelInfo.angStep;
	model->startAng = shapeModelInfo.startAng;
	model->endAng = shapeModelInfo.endAng;

	vector<Mat> imgPry;
	GetPyrImg(modImg, imgPry, shapeModelInfo.pyrNumber);

	for (int i = 0; i < imgPry.size(); i++)
	{
		//��ȡ����
		vector<Point> v_Coord_;
		ExtractModelContour(imgPry[i], shapeModelInfo, v_Coord_);
		if (v_Coord_.size() < 10)
			break;
		//��ȡģ���ݶ���Ϣ
		vector<Point2f> v_Coord, v_Grad;
		vector<float> v_Amplitude;
		ExtractModelInfo(imgPry[i], v_Coord_, v_Coord, v_Grad, v_Amplitude);
		if (v_Coord.size() < 10)
			break;
		//�������������
		ShapeInfo models;
		ReduceMatchPoint(v_Coord, v_Grad, v_Amplitude, models.coord, models.grad, shapeModelInfo.step);
		//��������
		GetContourGravity(models.coord, models.gravity);
		//���Ļ�����
		Point2f gravity = Point2f(-models.gravity.x, -models.gravity.y);
		TranContour(models.coord, gravity);
		//Mat colorImg;
		//cvtColor(imgPry[i], colorImg, COLOR_GRAY2BGR);
		//DrawContours(colorImg, models.coord, models.gravity);
		model->models.push_back(models);
		model->pyrNum++;
	}
	ComputeNMSRange(model->models[model->pyrNum - 1].coord, model->min_x, model->min_y);
	return true;
}
//========================================================================================

//ƥ��================================================================================================
void TopMatch(Mat &s_x, Mat &s_y, const vector<Point2f>& r_coord, const vector<Point2f>& r_grad,
	float minScore, float greediness, float angle, vector<MatchRes>& reses)
{
	vector<MatchRes> reses_;
	int maxW = s_x.cols - 2;
	int maxH = s_x.rows - 2;
	float NormGreediness = ((1 - greediness * minScore) / (1 - greediness)) / r_coord.size();
	float anMinScore = 1 - minScore;
	float NormMinScore = minScore / r_coord.size();

	for (int y = 2; y < maxH; ++y)
	{
		for (int x = 2; x < maxW; ++x)
		{
			float partial_score = 0.0f, score = 0.0f;
			int sum = 0.0;
			for (int index = 0; index < r_coord.size(); index++)
			{
				int cur_x = x + r_coord[index].x;
				int cur_y = y + r_coord[index].y;
				++sum;
				if (cur_x < 2 || cur_y < 2 || cur_x > maxW || cur_y > maxH)
					continue;
				short gx = s_x.at<short>(cur_y, cur_x);
				short gy = s_y.at<short>(cur_y, cur_x);
				if (abs(gx) > 5 || abs(gy) > 5)
				{
					float grad_x = 0.0f, grad_y = 0.0f;
					NormalGrad((int)gx, (int)gy, grad_x, grad_y);
					partial_score += (grad_x * r_grad[index].x + grad_y * r_grad[index].y);
					score = partial_score / sum;
					if (score < (min(anMinScore + NormGreediness * sum, NormMinScore * sum)))
						break;
				}
			}
			if (score > minScore)
			{
				MatchRes matchRes;
				matchRes.score = score;
				matchRes.c_x = x;
				matchRes.c_y = y;
				matchRes.angle = angle;
				reses.push_back(matchRes);
			}
		}
	}
}
void MatchShapeModel(const Mat &image, const vector<Point2f>& r_coord, const vector<Point2f>& r_grad,
	float minScore, float greediness, float angle, int *center, MatchRes &matchRes)
{
	int maxW = image.cols - 2;
	int maxH = image.rows - 2;

	float NormGreediness = ((1 - greediness * minScore) / (1 - greediness)) / r_coord.size();
	float anMinScore = 1 - minScore;
	float NormMinScore = minScore / r_coord.size();

	for (int y = center[1]; y < center[3]; y++)
	{
		for (int x = center[0]; x < center[2]; x++)
		{
			float partial_score = 0.0f, score = 0.0;
			int sum = 0.0;
			for (int index = 0; index < r_coord.size(); index++)
			{
				int cur_x = x + r_coord[index].x;
				int cur_y = y + r_coord[index].y;
				++sum;
				if (cur_x < 2 || cur_y < 2 || cur_x > maxW || cur_y > maxH)
					continue;

				int gx = 0, gy = 0;
				ComputeGrad(image, cur_x, cur_y, gx, gy);
				if (abs(gx) > 20 || abs(gy) > 20)
				{
					float grad_x = 0.0f, grad_y = 0.0f;
					NormalGrad(gx, gy, grad_x, grad_y);
					partial_score += (grad_x * r_grad[index].x + grad_y * r_grad[index].y);
					score = partial_score / sum;
					if (score < (min(anMinScore + NormGreediness * sum, NormMinScore * sum)))
						break;
				}
			}
			if (score > matchRes.score)
			{
				matchRes.score = score;
				matchRes.c_x = x;
				matchRes.c_y = y;
				matchRes.angle = angle;
			}
		}
	}
}
//====================================================================================================

//��������============================================================================================
void DrawShapeRes(Mat& image, ShapeInfo& models, vector<MatchRes>& res)
{
	for (int i = 0; i < res.size(); ++i)
	{
		vector<Point2f> r_coord, r_grad;
		RotateCoordGrad(models.coord, models.grad, r_coord, r_grad, res[i].angle);
		DrawContours(image, r_coord, Point2f(res[i].c_x, res[i].c_y));
	}
}
//====================================================================================================

//����ģ��===========================================================================================
void ClearModel(ShapeModel* &pModel)
{
	if (pModel != nullptr)
	{
		if (pModel->models.size() != 0)
		{
			pModel->models.clear();
		}
		delete pModel;
		pModel = nullptr;
	}
}
//===================================================================================================

//Ѱ��ģ��===========================================================================================
void FindShapeModel(Mat &srcImg, ShapeModel *model, vector<MatchRes> &MatchReses)
{
	if (MatchReses.size() > 0)
		MatchReses.clear();
	const int pyr_n = model->pyrNum - 1;
	vector<Mat> imgPry;
	GetPyrImg(srcImg, imgPry, pyr_n + 1);
	float angStep = model->angStep > 0.5 ? model->angStep : 0.5;
	float angleStep_ = angStep * pow(2, pyr_n + 1);

	int angNum = (model->endAng - model->startAng) / angleStep_ + 1;
	//����ƥ��
	Mat sobel_x, sobel_y;
	Sobel(imgPry[pyr_n], sobel_x, CV_16SC1, 1, 0, 3);
	Sobel(imgPry[pyr_n], sobel_y, CV_16SC1, 0, 1, 3);
	vector<vector<MatchRes>> mulMatchRes(angNum);

	int thread_count = 2 * omp_get_max_threads() / 3;
	thread_count = thread_count > angNum ? angNum : thread_count;
#pragma omp parallel for num_threads(thread_count)
	for (int i = 0; i < angNum; ++i)
	{
		vector<MatchRes> reses(0);
		float angle = model->startAng + i * angleStep_;
		vector<Point2f> r_coord, r_grad;
		RotateCoordGrad(model->models[pyr_n].coord, model->models[pyr_n].grad, r_coord, r_grad, angle);
		TopMatch(sobel_x, sobel_y, r_coord, r_grad, model->minScore, model->greediness, angle, reses);
		mulMatchRes[i] = reses;
	}

	//���зǼ���ֵ����
	vector<MatchRes> resNMS;
	ShapeNMS(mulMatchRes, resNMS, model->min_x, model->min_y, model->res_n);

	//Mat image0;
	//cvtColor(imgPry[pyr_n], image0, COLOR_GRAY2BGR);
	//DrawShapeRes(image0, model->models[pyr_n], resNMS);

	//������ƥ��
	//���������߳���
	thread_count = thread_count > 5 ? 5 : thread_count;
	vector<MatchRes> reses_(5);
	for (int k = 0; k < resNMS.size(); ++k)
	{
		for (int i = pyr_n - 1; i > -1; --i)
		{
			for (size_t j = 0; j < 5; ++j)
				reses_[j].score = 0.0f;
			angleStep_ = angStep * pow(2, i);
			float minScore = model->minScore / (i + 1);
			int center[4] = { 2 * resNMS[k].c_x - 10, 2 * resNMS[k].c_y - 10, 2 * resNMS[k].c_x + 10, 2 * resNMS[k].c_y + 10 };
#pragma omp parallel for num_threads(thread_count)
			for (int j = -2; j <= 2; ++j)
			{
				float angle = resNMS[k].angle + j * angleStep_;
				vector<Point2f> r_coord, r_grad;
				RotateCoordGrad(model->models[i].coord, model->models[i].grad, r_coord, r_grad, angle);
				MatchShapeModel(imgPry[i], r_coord, r_grad, minScore, model->greediness, angle, center, reses_[j+2]);
			}
			std::stable_sort(reses_.begin(), reses_.end());
			resNMS[k] = reses_[0];
		}
	}
	for (size_t i = 0; i < resNMS.size(); ++i)
	{
		if (resNMS[i].score > model->minScore)
			MatchReses.push_back(resNMS[i]);
	}

	Mat img;
	cvtColor(imgPry[0], img, COLOR_GRAY2BGR);
	DrawShapeRes(img, model->models[0], MatchReses);
	return;
}
//====================================================================================================

void shape_match_test()
{
	string imgPath = "D:/data/��λ����ͼƬ/test5.jpg";
	Mat modImg = imread(imgPath, 0);
	cv::Rect rect(1265, 857, 203, 212);
	Mat modeImg_ = modImg(rect).clone();
	ShapeModel *model = new ShapeModel;

	SPAPLEMODELINFO shapeModelInfo;
	shapeModelInfo.pyrNumber = 5;
	shapeModelInfo.lowVal = 100;
	shapeModelInfo.highVal = 200;
	shapeModelInfo.step = 3;

	CreateShapeModel(modeImg_, model, shapeModelInfo);
	vector<MatchRes> v_MatchRes;

	model->startAng = -180;
	model->endAng = 180;
	model->res_n = 9;
	model->minScore = 0.5;

	string testImgPath = "D:/data/��λ����ͼƬ/test5.jpg";
	Mat testImg = imread(testImgPath, 0);
	double el = cv::getTickCount();
	FindShapeModel(testImg, model, v_MatchRes);
	double e2 = cv::getTickCount();
	double time = (e2 - el) / cv::getTickFrequency() * 1000;
	cout << time << endl;
	delete model;
	model = nullptr;
	//Mat testImg = imread("5.png", 0);

	//MatchRes matchRes;
	//double t1 = getTickCount();
	//FindShapeModel(testImg, model, 0.5, -90, 90, 1, 0.9, matchRes);
	//double t = (getTickCount() - t1)/ getTickFrequency();
	//cout << t;
	return;
}