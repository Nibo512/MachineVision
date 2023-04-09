#pragma once
#include "../BaseOprFile/utils.h"
#include "../BaseOprFile/OpenCV_Utils.h"

/*˵����
	��Ϸ��̣�a * x^2 + b * x * y + c * y * y + d * x + e * y + f = 0
	pts��[in]ƽ���ϵĵ��
*/

//�㵽��Բ�ľ���--���򵥰棬���������==================================================
template <typename T>
double Img_PtsToEllipseDist(const T& pt, Ellipse2D& ellipse)
{
	double cosVal = std::cos(-ellipse.angle);
	double sinVal = std::sin(-ellipse.angle);
	double x_ = pt.x - ellipse.x;
	double y_ = pt.y - ellipse.y;
	double x = cosVal * x_ - sinVal * y_;
	double y = cosVal * y_ + sinVal * x_;
	double k = y / x;
	double a_2 = ellipse.a * ellipse.a;
	double b_2 = ellipse.b * ellipse.b;
	double coeff = a_2 * b_2 / (b_2 + a_2 * k * k);
	double x0 = -std::sqrt(coeff);
	double y0 = k * x0;
	double dist1 = std::sqrt(pow(x - x0, 2) + pow(y - y0, 2));

	x0 = std::sqrt(coeff);
	y0 = k * x0;
	double dist2 = std::sqrt(pow(x - x0, 2) + pow(y - y0, 2));
	double dist = dist1 < dist2 ? dist1 : dist2;
	return dist;
}
//======================================================================================

/*��Բ���̱�׼��*/
void Img_EllipseNormalization(vector<double>& ellipse_, Ellipse2D& normEllipse);

/*���һ�²����㷨������ԲԲ��
	inlinerPts��[in]���ڵ�
	thres��[in]��ֵ
*/
void Img_RANSACFitEllipse(NB_Array2D pts, Ellipse2D& ellipse, vector<int>& inliners, double thres);

/*��С���˷������Բ��
	weights��[in]Ȩ��
	ellipse��[out]
*/
void Img_OLSFitEllipse(NB_Array2D pts, vector<double>& weights, Ellipse2D& ellipse);

/*huber����Ȩ�أ�
	ellipse��[in]
	weights��[out]Ȩ��
*/
void Img_HuberEllipseWeights(NB_Array2D pts, Ellipse2D& ellipse, vector<double>& weights);

/*Tukey����Ȩ�أ�
	ellipse��[in]
	weights��[out]Ȩ��
*/
void Img_TukeyEllipseWeights(NB_Array2D pts, Ellipse2D& ellipse, vector<double>& weights);

/*�����Բ��
	ellipse��[out]
	k��[in]��������
	method��[in]��Ϸ�ʽ---��С���ˡ�huber��Tukey
*/
void Img_FitEllipse(NB_Array2D pts, Ellipse2D& ellipse, int k, NB_MODEL_FIT_METHOD method);


//��Բ��ϲ���
void Img_FitEllipseTest();

