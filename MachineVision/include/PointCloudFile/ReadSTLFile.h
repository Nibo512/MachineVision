#pragma once
#include "../BaseOprFile/utils.h"

//读取stl文件
bool ReadSTLFile(const string &fileName, PC_XYZ &pointcloud);

//读取ASCII文件
bool ReadSTLASCII(const string &filename);

//读取二值化文件
bool ReadSTLBinary(const string &filename, PC_XYZ &pointcloud);

//填充三角形
void FillTrangle(PC_XYZ &pt, PC_XYZ &dstPC);
