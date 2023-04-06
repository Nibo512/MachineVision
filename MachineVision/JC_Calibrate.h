#pragma once
#include "include/BaseOprFile/utils.h"

//��ȡ��������
void ReadWorldCoord(const string& filename, vector<vector<cv::Point3f>>& worldPts);

//��ȡ�������
void ReadCamCoord(const string& filename, vector<cv::Point3f>& camPts);

//�궨���1
float CablibCam_1(vector<vector<cv::Point3f>>& worldPts, vector<cv::Point3f>& camPts, vector<double> &transMat);

//�ĵ������任����
void GetTransMat(vector<cv::Point3f> &worldPts, vector<cv::Point3f> &camPts, cv::Mat &transMat);

//ȥ���Ļ�
void PtDecentration(vector<cv::Point3f>& srcPts, vector<cv::Point3f>& dstPts);

//����궨���
float CalError(vector<cv::Point3f>& worldPts, vector<cv::Point3f>& camPts, vector<double>& calibMat);

//��С���˷����任����AX = B
void LSMCalTransMat(vector<cv::Point3f> &worldPts, vector<cv::Point3f> &camPts, vector<double> &transMat);

//�������շ������������С���˷�
void LagrangeSolveTLS(vector<cv::Point3f> &worldPts, vector<cv::Point3f> &camPts, vector<double> &transMat, float thresVal);

//�궨���Գ���
void CalibTest();