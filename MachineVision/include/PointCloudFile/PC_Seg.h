#pragma once
#include "../BaseOprFile/utils.h"
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <pcl/features/boundary.h>
#include <pcl/segmentation/extract_clusters.h>

//���һ�²���ģ�ͷָ�
void PC_RANSACSeg(PC_XYZ &srcPC, PC_XYZ &dstPC, int mode, float thresVal);

/*����ŷʽ����ָ��*/
void PC_EuclideanSeg(PC_XYZ &srcPC, std::vector<P_IDX> &clusters, float distThresVal);

/*DBSCAN�ָ
	indexs��[out]����ĵ�������
	radius��[in]����������뾶
	n��[in]�����ĸ���---�����жϸõ��Ƿ�Ϊ���ĵ�
	minGroup��[in]��С���ĸ���
	maxGroup��[in]������ĸ���
*/
void PC_DBSCANSeg(PC_XYZ &srcPC, vector<vector<int>> &indexs, double radius, int n, int minGroup, int maxGroup);

/*DOG�ָ
	indexs��[out]����ĵ�������
	large_r��[in]��뾶
	small_r��[in]С�뾶
	thresVal��[in]��������֮��Ĳ�
*/
void PC_DONSeg(PC_XYZ &srcPC, vector<int> &indexs, double large_r, double small_r, double thresVal);

/*����ƽ��ָ
	plane��[in]�ο�ƽ��
	index��[out]����ĵ�������
	thresVal��[in]�㵽ƽ��ľ���
	orit��[in]����---0��ʾȡƽ���Ϸ��ĵ㡢1��ʾȡƽ���·��ĵ�
*/
void PC_SegBaseOnPlane(PC_XYZ &srcPC, Plane3D& plane, vector<int> &index, double thresVal, int orit);

/*�������ʷָ���Ʒָ
	normals��[in]���Ƶķ�����
	indexs��[out]����ĵ�������
	H_Thres��[in]����ֵ
	L_Thres��[in]����ֵ
*/
void PC_CurvatureSeg(PC_XYZ &srcPC, PC_N &normals, vector<int> &indexs, double H_Thres, double L_Thres);

/*���Ʒָ���Գ���*/
void PC_SegTest();