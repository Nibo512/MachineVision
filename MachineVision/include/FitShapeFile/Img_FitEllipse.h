#pragma once
#include "../BaseOprFile/utils.h"
#include "../BaseOprFile/OpenCV_Utils.h"

/*˵����
	��Ϸ��̣�a * x^2 + b * x * y + c * y * y + d * x + e * y + f = 0
	pts��[in]ƽ���ϵĵ��
*/

/*��Բ���̱�׼��*/
void Img_EllipseNormalization(vector<double>& ellipse_, Ellipse2D& normEllipse);

/*���һ�²����㷨������ԲԲ��
	inlinerPts��[in]���ڵ�
	thres��[in]��ֵ
*/
void Img_RANSACFitEllipse(NB_Array2D pts, Ellipse2D& ellipse, vector<int>& inliners, double thres);

/*��С���˷������Բ��
	weights��[in]Ȩ��
	ellipse��[out]
*/
void Img_OLSFitEllipse(NB_Array2D pts, vector<double>& weights, Ellipse2D& ellipse);

/*huber����Ȩ�أ�
	ellipse��[in]
	weights��[out]Ȩ��
*/
void Img_HuberEllipseWeights(NB_Array2D pts, Ellipse2D& ellipse, vector<double>& weights);

/*Tukey����Ȩ�أ�
	ellipse��[in]
	weights��[out]Ȩ��
*/
void Img_TukeyEllipseWeights(NB_Array2D pts, Ellipse2D& ellipse, vector<double>& weights);

/*�����Բ��
	ellipse��[out]
	k��[in]��������
	method��[in]��Ϸ�ʽ---��С���ˡ�huber��Tukey
*/
void Img_FitEllipse(NB_Array2D pts, Ellipse2D& ellipse, int k, NB_MODEL_FIT_METHOD method);


//��Բ��ϲ���
void Img_FitEllipseTest();

