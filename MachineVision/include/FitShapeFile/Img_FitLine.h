#pragma once
#include "../BaseOprFile/utils.h"
#include "../BaseOprFile/OpenCV_Utils.h"

/*˵����
	ֱ�߷��̣� a * x + b * y + c = 0
	pts��[in]ƽ���ϵĵ��
*/

/*���һ�²����㷨����ֱ��
	pts��[in]ƽ���ϵĵ��
	line��[out]�����Բ---[0]��x�����Ϸ�������
				[1]��y����ķ�������[2]��cֵ
	inliners��[in]���ڵ�
	thres��[in]��ֵ
*/
void Img_RANSACFitLine(NB_Array2D pts, Line2D& line, vector<int>& inliners, double thres);

/*��С���˷����ֱ�ߣ�
	weights��[in]Ȩ��
	line��[out]
*/
void Img_OLSFitLine(NB_Array2D pts, vector<double>& weights, Line2D& line);

/*Huber����Ȩ�أ�
	line��[in]
	weights��[out]Ȩ��
*/
void Img_HuberLineWeights(NB_Array2D pts, Line2D& line, vector<double>& weights);

/*Tukey����Ȩ�أ�
	line��[in]
	weights��[out]Ȩ��
*/
void Img_TukeyLineWeights(NB_Array2D pts, Line2D& line, vector<double>& weights);

/*ֱ����ϣ�
	line��[out]
	k��[in]��������
	method��[in]��Ϸ�ʽ---��С���ˡ�huber��turkey
*/
void Img_FitLine(NB_Array2D pts, Line2D& line, int k, NB_MODEL_FIT_METHOD method);

//��άֱ����ϲ���
void Img_FitLineTest();

