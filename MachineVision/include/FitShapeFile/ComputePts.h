#pragma once
#include "../BaseOprFile/utils.h"

//平面两线求点================================================================
template <typename T>
void Img_LineComputePt(cv::Vec3d& line1, cv::Vec3d& line2, T& pt)
{
	if (abs(line1[0] * line2[0] + line1[1] * line2[1]) > 1.0 - EPS)
		return;
	pt.x = (line1[1] * line2[2] - line2[1] * line1[2]) /
		(line2[1] * line1[0] - line1[1] * line2[0]);
	pt.y = (line1[0] * line2[2] - line2[0] * line1[2]) /
		(line2[0] * line1[1] - line1[0] * line2[1]);
}
//============================================================================

//三面共点====================================================================
template <typename T>
void PC_PlaneComputePt(cv::Vec4d& plane1, cv::Vec4d& plane2, cv::Vec4d& plane3, T& pt)
{
	double cosVal1 = plane1[0] * plane2[0] + plane1[1] * plane2[1] + plane1[2] * plane2[2];
	double cosVal2 = plane1[0] * plane3[0] + plane1[1] * plane3[1] + plane1[2] * plane3[2];
	double cosVal3 = plane2[0] * plane3[0] + plane2[1] * plane3[1] + plane2[2] * plane3[2];
	double eps_ = 1 - EPS;
	if (abs(cosVal1) > eps_ && abs(cosVal2) > eps_ && abs(cosVal3) > eps_)
		return;

	double e1 = plane2[2] * plane1[0] - plane1[2] * plane2[0];
	double f1 = plane2[2] * plane1[1] - plane1[2] * plane2[1];
	double g1 = -plane2[2] * plane1[3] + plane1[2] * plane2[3];

	double e2 = plane3[2] * plane2[0] - plane2[2] * plane3[0];
	double f2 = plane3[2] * plane2[1] - plane2[2] * plane3[1];
	double g2 = -plane3[2] * plane2[3] + plane2[2] * plane3[3];

	pt.y = (e2 * g1 - e1 * g2) / (e2 * f1 - e1 * f2);
	pt.x = (g1 - f1 * pt.y) / e1;
	pt.z = -(plane1[3] + plane1[0] * pt.x + plane1[1] * pt.y) / plane1[2];
}
//============================================================================

//空间两线距离最近的点========================================================
template <typename T1, typename T2>
double PC_LineNearestPt(T1 &line1, T1 &line2, T2 &pt1, T2 &pt2)
{
	float sum_a1 = 0.0f, sum_a2 = 0.0f, sum_a1a2 = 0.0f;
	float sum_m21a1 = 0.0f, sum_m21a2 = 0.0f;
	for (int i = 0; i < 3; ++i)
	{
		sum_a1 += line1[i] * line1[i];
		sum_a2 += line2[i] * line2[i];
		sum_a1a2 -= line1[i] * line2[i];
		sum_m21a1 += line1[i] * (line2[i + 3] - line1[i + 3]);
		sum_m21a2 += line2[i] * (line1[i + 3] - line2[i + 3]);
	}
	float deno = 1.0f / (sum_a1a2 * sum_a1a2 - sum_a1 * sum_a2);
	float t1 = (sum_a1a2 * sum_m21a2 - sum_a2 * sum_m21a1) * deno;
	float t2 = (sum_a1a2 * sum_m21a1 - sum_a1 * sum_m21a2) * deno;
	pt1 = { line1[0] * t1 + line1[3], line1[1] * t1 + line1[4], line1[2] * t1 + line1[5] };
	pt2 = { line2[0] * t2 + line2[3], line2[1] * t2 + line2[4], line2[2] * t2 + line2[5] };
	double diff_x = pt2.x - pt1.x;
	double diff_y = pt2.y - pt1.y;
	double diff_z = pt2.z - pt1.z;
	return diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;
}
//============================================================================

//点到平面的投影点============================================================
template <typename T1, typename T2, typename T3>
void PC_PtProjPlanePt(T1 &pt, T2 &plane, T3 &projPt)
{
	float dist = pt.x * plane[0] + pt.y * plane[1] + pt.z * plane[2] + plane[3];
	projPt.x = pt.x - dist * plane[0];
	projPt.y = pt.y - dist * plane[1];
	projPt.z = pt.z - dist * plane[2];
}
//============================================================================

//空间点到空间直线的投影======================================================
template <typename T1, typename T2, typename T3>
void PC_PtProjLinePt(T1 &pt, T2 &line, T3 &projPt)
{
	float scale = pt.x * line[0] + pt.y * line[1] + pt.z * line[2] -
		(line[3] * line[0] + line[4] * line[1] + line[5] * line[2]);
	projPt.x = line[3] + scale * line[0];
	projPt.y = line[4] + scale * line[1];
	projPt.z = line[5] + scale * line[2];
}
//============================================================================

//平面上点到直线的投影========================================================
template <typename T1, typename T2, typename T3>
void Img_PtProjLinePt(T1 &pt, T2 &line, T3 &projPt)
{
	float scale = pt.x * line[0] + pt.y * line[1] - (line[2] * line[0] + line[3] * line[1]);
	projPt.x = line[2] + scale * line[0]; projPt.y = line[3] + scale * line[1];
}
//============================================================================