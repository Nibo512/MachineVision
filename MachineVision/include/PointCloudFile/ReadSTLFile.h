#pragma once
#include "../BaseOprFile/utils.h"

//��ȡstl�ļ�
bool ReadSTLFile(const string &fileName, PC_XYZ &pointcloud);

//��ȡASCII�ļ�
bool ReadSTLASCII(const string &filename);

//��ȡ��ֵ���ļ�
bool ReadSTLBinary(const string &filename, PC_XYZ &pointcloud);

//���������
void FillTrangle(PC_XYZ &pt, PC_XYZ &dstPC);
