#pragma once
#include "utils.h"

//点到平面的距离==============================================================
template <typename T1>
double PC_PtToPlaneDist(T1 &pt, Plane3D &plane)
{
	return abs(pt.x * plane.a + pt.y * plane.b + pt.z * plane.c + plane.d);
}
//============================================================================

//向量归一化==================================================================
template <typename T>
void PC_VecNormal(T &p)
{
	double norm_ = 1 / std::max(std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z), EPS);
	p.x *= norm_; p.y *= norm_; p.z *= norm_;
}
//============================================================================

//空间点到空间直线的距离=====================================================
template <typename T>
double PC_PtToLineDist(T &pt, Line3D &line)
{
	double scale = pt.x * line.a + pt.y * line.b + pt.z * line.c -
		(line.x * line.a + line.y * line.b + line.z * line.c);
	double diff_x = line.x + scale * line.a - pt.x;
	double diff_y = line.y + scale * line.b - pt.y;
	double diff_z = line.z + scale * line.c - pt.z;
	return std::sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z);
}
//===========================================================================

//三维向量叉乘===============================================================
template <typename T1, typename T2, typename T3>
void PC_VecCross(T1 &vec1, T2 &vec2, T3 &vec, bool isNormal = true)
{
	vec.x = vec1.y * vec2.z - vec1.z * vec2.y;
	vec.y = vec1.z * vec2.x - vec1.x * vec2.z;
	vec.z = vec1.x * vec2.y - vec1.y * vec2.x;
	if (isNormal)
	{
		double norm_ = 1.0 / std::sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
		vec.x *= norm_; vec.y *= norm_; vec.z *= norm_;
	}
}
//=========================================================================

//计算点间距===============================================================
template <typename T1, typename T2>
double Img_ComputePPDist(T1 &pt1, T2 &pt2)
{
	double diff_x = pt1.x - pt2.x;
	double diff_y = pt1.y - pt2.y;
	return std::sqrt(max(diff_x * diff_x + diff_y * diff_y, EPS));
}
//=========================================================================

//点到圆或者球的距离=======================================================
template <typename T1, typename T2>
double PC_PtToCircleDist(T1& pt, T2& circle)
{
	double diff_x = pt.x - circle.x;
	double diff_y = pt.y - circle.y;
	double diff_z = pt.z - circle.z;
	return abs(std::sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z) - circle.r);
}
//=========================================================================

//罗格里格斯公式===========================================================
template <typename T>
void RodriguesFormula(T &rotAxis, double rotAng, cv::Mat &rotMat)
{
	if (rotMat.size() != cv::Size(3, 3))
		rotMat = cv::Mat(cv::Size(3, 3), CV_64FC1, cv::Scalar(0.0));
	double cosVal = std::cos(rotAng);
	double conVal_ = 1 - cosVal;
	double sinVal = std::sin(rotAng);
	double* pRotMat = rotMat.ptr<double>();

	pRotMat[0] = cosVal + rotAxis.x * rotAxis.x * conVal_;
	pRotMat[1] = rotAxis.x * rotAxis.y * conVal_ - rotAxis.z * sinVal;
	pRotMat[2] = rotAxis.x * rotAxis.z * conVal_ + rotAxis.y * sinVal;

	pRotMat[3] = rotAxis.y * rotAxis.x * conVal_ + rotAxis.z * sinVal;
	pRotMat[4] = cosVal + rotAxis.y * rotAxis.y * conVal_;
	pRotMat[5] = rotAxis.y * rotAxis.z * conVal_ - rotAxis.x * sinVal;

	pRotMat[6] = rotAxis.z * rotAxis.x * conVal_ - rotAxis.y * sinVal;
	pRotMat[7] = rotAxis.z * rotAxis.y * conVal_ + rotAxis.x * sinVal;
	pRotMat[8] = cosVal + rotAxis.z * rotAxis.z * conVal_;
}
//=========================================================================


