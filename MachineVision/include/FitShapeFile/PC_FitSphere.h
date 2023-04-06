#pragma once
#include "../BaseOprFile/utils.h"

/*˵����
	�򷽳̣�(x - a)^2 + (y - b)^2 + (z - c)^2 = r^2
	��Ϸ��̣�x^2 + y^2 + z^2 + A * x + B * y + c * z + d = 0
	pts��[in]�ռ��еĵ��
*/

/*���һ�²����㷨������
	inlinerPts��[in]���ڵ�
	thres��[in]��ֵ
*/
void PC_RANSACFitSphere(NB_Array3D pts, Sphere3D& sphere, vector<int>& inliners, double thres);

/*��С���˷������
	weights��[in]Ȩ��
	sphere��[out]�����
*/
void PC_OLSFitSphere(NB_Array3D pts, vector<double>& weights, Sphere3D& sphere);

/*Huber����Ȩ�أ�
	sphere��[in]
	weights��[out]Ȩ��
*/
void PC_HuberSphereWeights(NB_Array3D pts, Sphere3D& sphere, vector<double>& weights);

/*Tukey����Ȩ�أ�
	sphere��[in]
	weights��[out]Ȩ��
*/
void PC_TukeySphereWeights(NB_Array3D pts, Sphere3D& sphere, vector<double>& weights);

/*�����
	sphere��[out]
	k��[in]��������
	method��[in]��Ϸ�ʽ---��С���ˡ�huber��tukey
*/
void PC_FitSphere(NB_Array3D pts, Sphere3D& sphere, int k, NB_MODEL_FIT_METHOD method);

//�ռ�����ϲ���
void PC_FitSphereTest();
