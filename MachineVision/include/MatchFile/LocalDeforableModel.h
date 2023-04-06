#pragma once
#include "../BaseOprFile/utils.h"
#include "ShapeModelBase.h"

struct LocalDeforModelInfo
{
	vector<Point2f> coord;
	vector<Point2f> grad;
	Point2f gravity;
	vector<Point2f> normals_;
	vector<vector<int>> segContIdx;
	vector<int> segContMapIdx;
	LocalDeforModelInfo():gravity(0,0)
	{
		coord.clear();
		grad.clear();
		normals_.clear();
		segContIdx.clear();
		segContMapIdx.clear();
	}
};

struct LocalDeforModel
{
	int pyrNum;
	double startAng;
	double endAng;
	double angStep;
	double minScale;
	double maxScale;
	double minScore;
	double greediness;
	int transLen;
	vector<LocalDeforModelInfo> models;

	LocalDeforModel() :pyrNum(0), startAng(0.0), endAng(0.0),
		angStep(0.0), minScale(1.0), maxScale(1.0),
		minScore(0.5), greediness(0.9), transLen(2)
	{
		models.resize(0);
	}
};

struct LocalMatchRes : public MatchRes
{
	vector<int> translates;
	vector<bool> flags;
	vector<Point> gravitys;
	LocalMatchRes()
	{
		translates.clear();
		flags.clear();
		gravitys.clear();
	}
	bool operator<(const LocalMatchRes &other) const
	{
		return score > other.score;
	}
};

//����ģ��
void CreateLocalDeforableModel(Mat &modImg, LocalDeforModel* &model, SPAPLEMODELINFO &shapeModelInfo);

//��ȡ����
void ExtractModelContour(Mat &srcImg, SPAPLEMODELINFO &shapeModelInfo, vector<vector<Point>> &contours);

//ģ������
void GetKNearestPoint(vector<Point2f> &contours, vector<Point2f> &grads, LocalDeforModelInfo &localDeforModelInfo, int ptNum);

//�����������ķ�������
void ComputeSegContourVec(LocalDeforModel &model);

//�������Ļ�ȡÿ��С������ӳ������
void GetMapIndex(LocalDeforModel& localDeforModel);

//������߲��ƽ�ƾ���
void ComputeTopTransLen(LocalDeforModel& localDeforModel);

//ƽ������
void TranslationContour(const vector<Point2f>& contour, const vector<int>& contIdx,
	const Point3f& normals, vector<Point2f>& tranContour, int transLen);

//����ƥ��
void TopMatch(const Mat &s_x, const Mat &s_y, const vector<Point2f>& r_coord, const vector<Point2f>& r_grad, const vector<vector<int>>& segIdx,
	const vector<Point3f>& normals_, double minScore, double angle, int transLenP, LocalMatchRes& reses);

//ƥ��
void Match(const Mat &image, const vector<Point2f>& r_coord, const vector<Point2f>& r_grad, const vector<vector<int>>& segIdx, const vector<Point2f>& normals_,
	int* center, double minScore, double angle, vector<int>& transLen_down, vector<bool>& contourFlags, /*vector<Point>& gravitys, */LocalMatchRes& reses);

//��ȡƽ����
void GetTranslation(vector<int>& segContMapIdx, LocalMatchRes& res, vector<int>& transLen_down,
	vector<bool>& contourFlags, vector<Point>& gravitys);

//��ת��������
void RotContourVec(const vector<Point2f>& srcVec, vector<Point2f>& dstVec, double rotAng);

//����ƥ�䵽�Ľ��
void DrawLocDeforRes(Mat& image, LocalDeforModelInfo& models, LocalMatchRes& res, vector<bool>& contourFlags);

//ƥ��
void LocalDeforModelMatch(Mat &modImg, LocalDeforModel* &model);

void LocalDeforModelTest();