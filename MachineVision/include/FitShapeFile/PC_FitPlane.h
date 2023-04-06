#pragma once
#include "../BaseOprFile/utils.h"

/*˵����
	ƽ�淽�̣�a * x + b * y + c * z + d = 0
	pts��[in]�ռ��еĵ��
*/

/*���һ�²����㷨����ƽ�棺
	inlinerPts��[in]���ڵ�
	thres��[in]��ֵ
*/
void PC_RANSACFitPlane(NB_Array3D pts, Plane3D& plane, vector<int>& inliners, double thres);

/*��С���˷����ƽ�棺
	weights��[in]Ȩ��
	plane��[out]
*/
void PC_OLSFitPlane(NB_Array3D pts, vector<double>& weights, Plane3D& plane);

/*huber����Ȩ�أ�
	plane��[in]
	weights��[out]Ȩ��
*/
void PC_HuberPlaneWeights(NB_Array3D pts, Plane3D& plane, vector<double>& weights);

/*Tukey����Ȩ�أ�
	plane��[in]
	weights��[out]Ȩ��
*/
void PC_TukeyPlaneWeights(NB_Array3D pts, Plane3D& plane, vector<double>& weights);

/*ƽ����ϣ�
	plane��[out]
	k��[in]��������
	method��[in]��Ϸ�ʽ---��С���ˡ�huber��turkey
*/
void PC_FitPlane(NB_Array3D pts, Plane3D& plane, int k, NB_MODEL_FIT_METHOD method);

//�ռ�ƽ����ϲ���
void PC_FitPlaneTest();
