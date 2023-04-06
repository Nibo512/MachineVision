#pragma once
#include "utils.h"

/*��״�任��
	pc��[in--out]���������״
	shape��[in]��״����
	mode��[in] 0--��ʾֱ�ߡ�1--��ʾƽ��
*/
void PC_ShapeTrans(PC_XYZ::Ptr& pc, cv::Vec6d& shape, cv::Point3d& vec);

/*����ֱ�ߣ�
	linePC��[out]���ֱ��
	length��[in]ֱ�߳���
	line��[in]ֱ�߲���
	step��[in]����
*/
void PC_DrawLine(PC_XYZ::Ptr& linePC, cv::Vec6d& line, double length, double step);

/*����ƽ�棺
	planePC��[out]���ƽ��
	length��[in]ƽ�泤
	width��[in]ƽ���
	plane��[in]ƽ�����
	step��[in]����
*/
void PC_DrawPlane(PC_XYZ::Ptr& planePC, cv::Vec6d& plane, double length, double width, double step);

/*������:
	spherePC��[out]�������
	center��[in]����
	raduis��[in]�뾶
	step��[in]�ǶȲ���
*/
void PC_DrawSphere(PC_XYZ::Ptr& spherePC, P_XYZ& center, double raduis, double step);

/*���������棺
	ellipsoidPC��[out]�����������
	center��[in]���������λ��
	a��b��c��[in]�ֱ�Ϊx��y��z����᳤
	step��[in]�ǶȲ���
*/
void PC_DrawEllipsoid(PC_XYZ::Ptr& ellipsoidPC, cv::Vec6d& ellipsoid, double a, double b, double c, double step);

/*������Բ��
	ellipseImg��[out]�������Բ
	center��[in]��Բ������λ��
	a��b��[in]�ֱ�Ϊx��y�������᳤
	rotAng��[in]��ת�Ƕ�
	step��[in]�ǶȲ���
*/
void Img_DrawEllipse(Mat& ellipseImg, cv::Point2d& center, double rotAng, double a, double b, double step);

/*���������壨���ģ���
	rectPC��[out]���ƽ��
	cube��[in]���������
	step��[in]����
*/
void PC_DrawCube(PC_XYZ::Ptr& rectPC, cv::Vec6d& cube, double a, double b, double c, double step);

/*���ƿռ�԰��
	circlePC��[out]���ƽ��
	circle��[in]�ռ�԰����
	r��[in]�ռ�԰�뾶
	step��[in]����
*/
void PC_DrawCircle(PC_XYZ::Ptr& circlePC, cv::Vec6d& circle, double r, double step);

/*���������
	srcPC��[in]ԭʼ����
	noisePC��[out]��������
	range��[in]������С
	step��[in]���Ʋ���
*/
void PC_AddNoise(PC_XYZ::Ptr& srcPC, PC_XYZ::Ptr& noisePC, int range, int step);

//���Գ���
void DrawShapeTest();