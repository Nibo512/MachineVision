#pragma once
#include "../BaseOprFile/utils.h"
#include "opencv2/features2d.hpp"

//ͼ��ת�Ҷ�ͼ
void ImageToGray(Mat& srcImg, Mat& grayImg);

//��ȡSIFT�ǵ�
void ExtractSiftPt(Mat& srcImg, Mat& tarImg, vector<KeyPoint>& srcPts, 
	vector<KeyPoint>& tarPts, Mat& desSrc, Mat& desTar, int pts_num);

//���������--ͨ��DMatch
void KeyPtsMatch(Mat& desSrc, Mat& desTar, vector<DMatch>& good_matches);

//���������--ͨ��Point2f
void KeyPtsMatch_Pts(vector<KeyPoint>& srcKeyPts,vector<KeyPoint>& tarKeyPts, 
	Mat& desSrc, Mat& desTar, vector<Point2f>& srcPts, vector<Point2f>& tarPts);

//RANSAC�㷨ѡ��ɿ���ƥ���
void SelectGoodMatchPts(vector<KeyPoint>& srcPts, vector<KeyPoint>& tarPts, vector<DMatch>& good_matches,
	vector<Point2f>& obj, vector<Point2f>& scene, Mat& tranMat, vector<uchar>& inliers);

//��ȡ�ڵ�--ͨ��DMatch
void ExtractInlinerPts(vector<DMatch>& good_matches, vector<uchar>& inliers, vector<DMatch>& inlinerPts);

//��ȡ�ڵ�--ͨ��Point2f
void ExtractInlinerPts_Pts(vector<Point2f>& srcPts, vector<Point2f>& tarPts, vector<uchar>& inliers, 
	vector<Point2f>& srcPts_o, vector<Point2f>& tarPts_o);

//��С���˷��Ż�:AX = B
void SiftLSOpt(vector<Point2f>& srcPts, vector<Point2f>& tarPts, Mat& tranMat);

//��ȡSIFT�ǵ�
void SiftPtTest();
