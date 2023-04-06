#pragma once
#include "../BaseOprFile/utils.h"

//������Ƶ���С��Χ��
void PC_ComputeOBB(const PC_XYZ &srcPC, PC_XYZ &obb);

/*��ȡ���ƣ�
	indexes��[in]��������
	dstPC��[out]����ĵ���
*/
void PC_ExtractPC(const PC_XYZ &srcPC, vector<int> &indexes, PC_XYZ &dstPC);

/*����ֱ��ͶӰƽ����
	dstPC��[out]����ĵ���
	size��[in]��������ڵ�ĸ���
	thresVal��[in]��������ڵ�ĸ���
*/
void PC_LineProjSmooth(const PC_XYZ &srcPC, PC_XYZ &dstPC, int size, double thresVal);

//������Ƶ�����
void PC_GetPCGravity(PC_XYZ &srcPC, P_XYZ &gravity);

//������ͶӰ��XY��άƽ��
void PC_ProjectToXY(PC_XYZ &srcPC, cv::Mat &xyPlane);

//������Ƶķ�����
void PC_ComputePCNormal(PC_XYZ &srcPC, PC_N &normals, float radius);


