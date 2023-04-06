#pragma once
#include "../BaseOprFile/utils.h"
#include "../BaseOprFile/OpenCV_Utils.h"

/*˵����
	԰���̣�(x - a)^2 + (y - b)^2 = r^2
	��Ϸ��̣�x^2 + y^2 + A * x + B * y + C = 0
	pts��[in]ƽ���ϵĵ��
*/

/*���һ�²����㷨����Բ��
	inlinerPts��[in]���ڵ�
	thres��[in]��ֵ
*/
void Img_RANSACFitCircle(NB_Array2D pts, Circle2D& circle, vector<int>& inliners, double thres);

/*��С���˷����԰��
	weights��[in]Ȩ��
	circle��[out]
*/
void Img_OLSFitCircle(NB_Array2D pts, vector<double>& weights, Circle2D& circle);

/*huber����Ȩ�أ�
	circle��[in]
	weights��[out]Ȩ��
*/
void Img_HuberCircleWeights(NB_Array2D pts, Circle2D& circle, vector<double>& weights);

/*Tukey����Ȩ�أ�
	circle��[in]
	weights��[out]Ȩ��
*/
void Img_TukeyCircleWeights(NB_Array2D pts, Circle2D& circle, vector<double>& weights);

/*���԰
	circle��[out]
	k��[in]��������
	method��[in]��Ϸ�ʽ---��С���ˡ�huber��Tukey
*/
void Img_FitCircle(NB_Array2D pts, Circle2D& circle, int k, NB_MODEL_FIT_METHOD method);

//��άԲ��ϲ���
void Img_FitCircleTest();
