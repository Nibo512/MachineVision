#pragma once
#include "../BaseOprFile/utils.h"

//计算法向量以及重心
void CalNormalAndGravity(PC_XYZ& srcPC, Eigen::Vector3f& normals, P_XYZ& gravity);

//计算局部转换矩阵
void CalLocCoordSys(Eigen::Vector3f& normals, P_XYZ& gravity, Eigen::Matrix4f& locTransMat);

//计算权重
void CalPtWeigth(PC_XYZ& pts, vector<float>& w);

//计算多项式
void CalPolyVec(P_XYZ& pt, vector<float>& polyVec, int order);

//计算矩阵a、b
void CalMatAB(vector<float>& polyVec, P_XYZ& pt, Eigen::MatrixXf& a_mat,
	Eigen::MatrixXf& b_mat, int order, float w_);

//计算系数矩阵
void CalCoffMat(PC_XYZ& srcPC, Eigen::MatrixXf& coffMat, vector<float>& ws, int order);

//投影
void PtProjToPoly(P_XYZ& pt, Eigen::MatrixXf& coffMat, P_XYZ& projPt, int order);

//MLS平滑
void MLSSmooth(PC_XYZ &srcPC, PC_XYZ &dstPC, float radius, int order);

//测试MLS
void TestMLS();

