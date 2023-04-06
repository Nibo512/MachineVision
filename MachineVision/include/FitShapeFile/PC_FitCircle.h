#pragma once
#include "../BaseOprFile/utils.h"

/*˵����
	�ռ�԰������һ�㷽�̣�x^2 + y^2 + z^2 + A * x + B * y + c * z + d = 0
				���߷���(x - a)*vx + (y - b)*vy + (z - c)*vz = 0
				vx��vy��vzΪԲ����ƽ��ķ�����
	pts��[in]�ռ��еĵ��
*/

/*���һ�²����㷨����Բ��
	inlinerPts��[in]���ڵ�
	thres��[in]��ֵ
*/
void PC_RANSACFitCircle(NB_Array3D pts, Circle3D& circle, vector<int>& inliners, double thres);

/*��С���˷���Ͽռ�ռ�Բ��
	weights��[in]Ȩ��
	circle��[out]
*/
void PC_OLSFit3DCircle(NB_Array3D pts, vector<double>& weights, Circle3D& circle);

/*��С���˷����Բ��
	circle��[out]���Բ
	weights��[in]Ȩ��
*/
void PC_HuberCircleWeights(NB_Array3D pts, Circle3D& circle, vector<double>& weights);

/*Tukey����Ȩ�أ�
	circle��[out]���Բ
	weights��[in]Ȩ��
*/
void PC_TukeyCircleWeights(NB_Array3D pts, Circle3D& circle, vector<double>& weights);

/*���Բ��
	circle��[out]
	k��[in]��������
	method��[in]��Ϸ�ʽ---��С���ˡ�huber��tukey
*/
void PC_FitCircle(NB_Array3D pts, Circle3D& circle, int k, NB_MODEL_FIT_METHOD method);

//�ռ���άԲ��ϲ���
void PC_FitCircleTest();

