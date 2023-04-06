#pragma once
#include "../BaseOprFile/utils.h"
#include "ContourOpr.h"
#include "ShapeModelBase.h"
#include <map>

//ģ����Ϣ
struct ShapeInfo
{
	vector<Point2f> coord;
	vector<Point2f> grad;
	Point2f gravity;
	ShapeInfo() :gravity(Point2f(0.0f, 0.0f))
	{
		coord.clear();
		grad.clear();
	}
};

//���в�����ģ����Ϣ
struct ShapeModel
{
	int pyrNum;
	float startAng;
	float endAng;
	float angStep;
	float minScore;
	float greediness;
	int res_n;       //ƥ�����
	int min_x;     //�Ǽ���ֵ���Ƶ�x��Χ 
	int min_y;     //�Ǽ���ֵ ����y��Χ
	vector<ShapeInfo> models;
	ShapeModel() :pyrNum(0), startAng(0.0f), endAng(0.0f), angStep(0.0f),
		minScore(0.5f), greediness(0.9f), res_n(0), min_x(0), min_y(0)
	{
		models.resize(0);
	}
};

//����ģ��
bool CreateShapeModel(Mat &modImg, ShapeModel* &model, SPAPLEMODELINFO &shapeModelInfo);

//Ѱ��ģ��
void FindShapeModel(Mat &srcImg, ShapeModel *model, vector<MatchRes> &MatchReses);

//����ƥ��
void TopMatch(Mat &s_x, Mat &s_y, const vector<Point2f>& r_coord, const vector<Point2f>& r_grad,
	float minScore, float greediness, float angle, vector<MatchRes>& reses);

//ƥ��
void MatchShapeModel(const Mat &image, const vector<Point2f>& r_coord, const vector<Point2f>& r_grad,
	float minScore, float greediness, float angle, int *center, MatchRes &matchRes);

//��������
void DrawShapeRes(Mat& image, ShapeInfo& models, vector<MatchRes>& res);

//�ͷ�ģ��
void ClearModel(ShapeModel* &pModel);

//���Գ���
void shape_match_test();