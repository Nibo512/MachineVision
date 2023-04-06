#pragma once
#include "../BaseOprFile/utils.h"

/*˵����
	ֱ�߷��̣� x = a * t + x0; y = b * t + y0; z = c * t + z0
	pts��[in]�ռ��еĵ��
*/

/*���һ�²����㷨����ռ�ֱ�ߣ�
	inlinerPts��[in]���ڵ�
	thres��[in]��ֵ
*/
void PC_RANSACFitLine(NB_Array3D pts, Line3D& line, vector<int>& inlinerPts, double thres);

/*��С���˷���Ͽռ�ֱ�ߣ�
	weights��[in]Ȩ��
	line��[out]
*/
//template <typename T1, typename T2>
void PC_OLSFit3DLine(NB_Array3D pts, vector<double>& weights, Line3D& line);

/*Huber����Ȩ�أ�
	line��[in]
	weights��[out]Ȩ��
*/
//template <typename T1, typename T2>
void PC_Huber3DLineWeights(NB_Array3D pts, Line3D& line, vector<double>& weights);

/*Tukey����Ȩ�أ�
	line��[in]
	weights��[out]Ȩ��
*/
//template <typename T1, typename T2>
void PC_Tukey3DLineWeights(NB_Array3D pts, Line3D& line, vector<double>& weights);

/*�ռ�ֱ����ϣ�
	line��[out]
	k��[in]��������
	method��[in]��Ϸ�ʽ---��С���ˡ�huber��turkey
*/
//template <typename T1, typename T2>
void PC_Fit3DLine(NB_Array3D pts, Line3D& line, int k, NB_MODEL_FIT_METHOD method);

//�ռ���λֱ����ϲ���
void PC_FitLineTest();
