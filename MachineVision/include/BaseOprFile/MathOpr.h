#pragma once
#include "utils.h"

//点到直线的投影==============================================================
template <typename T1, typename T2>
void Pt3DProjToLine(const T1& src, Line3D& line, T2& dst)
{
	float scale = src.x * line.a + src.y * line.b + src.z * line.c -
		(line.x * line.a + line.y * line.b + line.z * line.c);
	dst.x = line.x + scale * line.a;
	dst.y = line.y + scale * line.b;
	dst.z = line.z + scale * line.c;
}
template <typename T1, typename T2>
void Pt2DProjToLine(T1& ref_p, T2& line, T1& proj_p)
{
	float scale = ref_p.x * line[0] + ref_p.y * line[1] -
		(line[2] * line[0] + line[3] * line[1]);
	proj_p.x = line[2] + scale * line[0]; proj_p.y = line[3] + scale * line[1];
}
//============================================================================

//两直线求交点================================================================
template <typename T1, typename T2>
float PC_TwoLinesNearestPt(Line3D& line1, Line3D& line2, T2& pt1, T2& pt2)
{
	float sum_a1 = line1.a * line1.a + line1.b * line1.b + line1.c * line1.c;
	float sum_a2 = line2.a * line2.a + line2.b * line2.b + line2.c * line2.c;
	float sum_a1a2 = -(line1.a * line2.a + line1.b * line2.b + line1.c * line2.c);

	float diff_x = line2.x - line1.x;
	float diff_y = line2.y - line1.y;
	float diff_z = line2.z - line1.z;
	float sum_m21a1 = line1.a * diff_x + line1.b * diff_y + line1.c * diff_z;
	float sum_m21a2 = -(line2.a * diff_x + line2.b * diff_y + line2.c * diff_z);

	float deno = 1.0f / (sum_a1a2 * sum_a1a2 - sum_a1 * sum_a2);
	float t1 = (sum_a1a2 * sum_m21a2 - sum_a2 * sum_m21a1) * deno;
	float t2 = (sum_a1a2 * sum_m21a1 - sum_a1 * sum_m21a2) * deno;
	pt1 = { line1.a * t1 + line1.x, line1.b * t1 + line1.y, line1.c * t1 + line1.z };
	pt2 = { line2.a * t2 + line2.x, line2.b * t2 + line2.b, line2.c * t2 + line2.z };

	float diff_x = pt2.x - pt1.x;
	float diff_y = pt2.y - pt1.y;
	float diff_z = pt2.z - pt1.z;
	return diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;
}
//============================================================================

//点到平面的投影==============================================================
template <typename T1, typename T2>
void PtProjToPlane(T1& pt, Plane3D& plane, T1& projPt)
{
	float dist = pt.x * plane.a + pt.y * plane.b + pt.z * plane.c + plane.d;
	projPt.x = pt.x - plane.a * dist;
	projPt.y = pt.y - plane.b * dist;
	projPt.z = pt.z - plane.c * dist;
}
//============================================================================

//点到平面的距离==============================================================
template <typename T1>
double PC_PtToPlaneDist(T1 &pt, Plane3D &plane)
{
	return abs(pt.x * plane.a + pt.y * plane.b + pt.z * plane.c + plane.d);
}
//============================================================================

//三面共点====================================================================
template <typename T>
void PC_3PlanesConPt(Plane3D plane1, Plane3D plane2, Plane3D plane3, T& point)
{
	float e1 = plane2.c * plane1.a - plane1.c * plane2.a;
	float f1 = plane2.c * plane1.b - plane1.c * plane2.b;
	float g1 = -plane2.c * plane1.d + plane1.c * plane2.d;

	float e2 = plane3.c * plane2.a - plane2.c * plane3.a;
	float f2 = plane3.c * plane2.b - plane2.c * plane3.b;
	float g2 = -plane3.c * plane2.d + plane2.c * plane3.d;

	point.y = (e2 * g1 - e1 * g2) / (e2 * f1 - e1 * f2);
	point.x = (g1 - f1 * point.y) / e1;
	point.z = -(plane1.d + plane1.a * point.x + plane1.b * point.y) / plane1.c;
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
double PC_PtToLineDist(T& pt, Line3D &line)
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

//两点距离=================================================================
template <typename T1, typename T2>
double PC_PPDist(T1 &pt1, T2 &pt2)
{
	float diff_x = pt2.x - pt1.x;
	float diff_y = pt2.y - pt1.y;
	float diff_z = pt2.z - pt1.z;
	return std::sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z);
}
//=========================================================================

//计算两点间的向量=========================================================
template <typename T1, typename T2>
void PC_PPVec(T1& pt1, T2& pt2, P_XYZ& normal)
{
	float diff_x = pt2.x - pt1.x;
	float diff_y = pt2.y - pt1.y;
	float diff_z = pt2.z - pt1.z;
	float norm_ = 1.0f / std::sqrt(std::max(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z, 1e-8f));
	normal = { diff_x * norm_, diff_y * norm_, diff_z * norm_ };
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

//点的变换=================================================================
template <typename T1, typename T2>
void PC_TransLatePoint(Eigen::Matrix4f& transMat, T1& pt, T2& pt_t)
{
	float x = transMat(0, 0) * pt.x + transMat(0, 1) * pt.y + transMat(0, 2) * pt.z + transMat(0, 3);
	float y = transMat(1, 0) * pt.x + transMat(1, 1) * pt.y + transMat(1, 2) * pt.z + transMat(1, 3);
	float z = transMat(2, 0) * pt.x + transMat(2, 1) * pt.y + transMat(2, 2) * pt.z + transMat(2, 3);
	pt_t = { x,y,z };
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


