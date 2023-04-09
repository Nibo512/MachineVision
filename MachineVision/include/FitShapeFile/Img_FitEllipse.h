#pragma once
#include "../BaseOprFile/utils.h"
#include "../BaseOprFile/OpenCV_Utils.h"

/*说明：
	拟合方程：a * x^2 + b * x * y + c * y * y + d * x + e * y + f = 0
	pts：[in]平面上的点簇
*/

//点到椭圆的距离--超简单版，不建议采用==================================================
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

/*椭圆方程标准化*/
void Img_EllipseNormalization(vector<double>& ellipse_, Ellipse2D& normEllipse);

/*随机一致采样算法计算椭圆圆：
	inlinerPts：[in]局内点
	thres：[in]阈值
*/
void Img_RANSACFitEllipse(NB_Array2D pts, Ellipse2D& ellipse, vector<int>& inliners, double thres);

/*最小二乘法拟合椭圆：
	weights：[in]权重
	ellipse：[out]
*/
void Img_OLSFitEllipse(NB_Array2D pts, vector<double>& weights, Ellipse2D& ellipse);

/*huber计算权重：
	ellipse：[in]
	weights：[out]权重
*/
void Img_HuberEllipseWeights(NB_Array2D pts, Ellipse2D& ellipse, vector<double>& weights);

/*Tukey计算权重：
	ellipse：[in]
	weights：[out]权重
*/
void Img_TukeyEllipseWeights(NB_Array2D pts, Ellipse2D& ellipse, vector<double>& weights);

/*拟合椭圆：
	ellipse：[out]
	k：[in]迭代次数
	method：[in]拟合方式---最小二乘、huber、Tukey
*/
void Img_FitEllipse(NB_Array2D pts, Ellipse2D& ellipse, int k, NB_MODEL_FIT_METHOD method);


//椭圆拟合测试
void Img_FitEllipseTest();

