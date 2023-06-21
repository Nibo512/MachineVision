#pragma once
#include "../BaseOprFile/utils.h"

//���㷨�����Լ�����
void CalNormalAndGravity(PC_XYZ& srcPC, Eigen::Vector3f& normals, P_XYZ& gravity);

//����ֲ�ת������
void CalLocCoordSys(Eigen::Vector3f& normals, P_XYZ& gravity, Eigen::Matrix4f& locTransMat);

//����Ȩ��
void CalPtWeigth(PC_XYZ& pts, vector<float>& w);

//�������ʽ
void CalPolyVec(P_XYZ& pt, vector<float>& polyVec, int order);

//�������a��b
void CalMatAB(vector<float>& polyVec, P_XYZ& pt, Eigen::MatrixXf& a_mat,
	Eigen::MatrixXf& b_mat, int order, float w_);

//����ϵ������
void CalCoffMat(PC_XYZ& srcPC, Eigen::MatrixXf& coffMat, vector<float>& ws, int order);

//ͶӰ
void PtProjToPoly(P_XYZ& pt, Eigen::MatrixXf& coffMat, P_XYZ& projPt, int order);

//MLSƽ��
void MLSSmooth(PC_XYZ &srcPC, PC_XYZ &dstPC, float radius, int order);

//����MLS
void TestMLS();

