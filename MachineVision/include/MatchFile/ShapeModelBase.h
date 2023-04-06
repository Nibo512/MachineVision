#pragma once
#include "../BaseOprFile/utils.h"

struct SPAPLEMODELINFO
{
	int pyrNumber;       // ����������
	int minContourLen;   //��������С����
	int maxContourLen;  //��������󳤶�
	int lowVal;    //������ȡ����ֵ
	int highVal;   //������ȡ����ֵ
	int extContouMode;  //������ȡģʽ
	int step;   //ѡ�㲽��
	double startAng;
	double endAng;
	double angStep;
	double minScale;
	double maxScale;
	SPAPLEMODELINFO() :pyrNumber(1), minContourLen(0),
		maxContourLen(1e9),	lowVal(15), highVal(30),
		step(3), extContouMode(0), minScale(1.0), maxScale(1.0)
	{}
};

//������
struct MatchRes
{
	int c_x;  //x����
	int c_y;  //y_����
	float angle;
	float score;
	MatchRes() :c_x(0), c_y(0), angle(0.0f), score(0.0f)
	{ }
	bool operator<(const MatchRes& res) const
	{
		return this->score > res.score;
	}
	void reset()
	{
		this->c_x = 0;
		this->c_y = 0;
		this->angle = 0.0f;
		this->score = 0.0f;
	}
};

//����Ǽ���ֵ���Ƶĳ���
void  ComputeNMSRange(vector<Point2f>& contour, int& min_x, int& min_y);

//����ͼ�������
void GetPyrImg(Mat &srcImg, vector<Mat> &pyrImg, int pyrNumber);

//��ȡģ�������
void ExtractModelContour(Mat &srcImg, SPAPLEMODELINFO &shapeModelInfo, vector<Point> &contour);

//��ȡģ����Ϣ
void ExtractModelInfo(Mat &srcImg, vector<Point> &contour, vector<Point2f> &v_Coord, vector<Point2f> &v_Grad, vector<float> &v_Amplitude);

//��һ���ݶ�
void NormalGrad(int grad_x, int grad_y, float &grad_xn, float &grad_yn);

//�Ǽ���ֵ����
void ShapeNMS(vector<vector<MatchRes>> &MatchReses, vector<MatchRes> &nmsRes, int x_min, int y_min, int matchNum);

//�����ݶ�
void ComputeGrad(const Mat &srcImg, int idx_x, int idx_y, int& grad_x, int& grad_y);

//������ת�Լ��ݶ�
void RotateCoordGrad(const vector<Point2f> &coord, const vector<Point2f> &grad,
	vector<Point2f> &r_coord, vector<Point2f> &r_grad, float rotAng);

//������������
void DrawContours(Mat &srcImg, vector<Point2f> &v_Coord, Point2f offset);

//����ƥ������
void ReduceMatchPoint(vector<Point2f> &v_Coord, vector<Point2f> &v_Grad, vector<float> &v_Amplitude,
	vector<Point2f> &v_RedCoord, vector<Point2f> &v_RedGrad, int step = 3);