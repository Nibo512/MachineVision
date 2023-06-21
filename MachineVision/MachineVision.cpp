// MachineVision.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include "include/BaseOprFile/utils.h"
#include "include/BaseOprFile/OpenCV_Utils.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cusolverDn.h>
#include "include/PointCloudFile/PC_Filter.h"
#include "include/PointCloudFile/PointCloudOpr.h"
#include <Eigen/Eigenvalues>
#include <cublas_v2.h>
#include "include/PointCloudMatch/CPDMatch/NonRigidCPD.h"
#include "include/BaseOprFile/DrawShape.h"
#include "include/FitShapeFile/PC_FitCircle.h"
#include "include/FitShapeFile/PC_FitSphere.h"
#include "include/PointCloudFile/MLSSmooth.h"
#include "include/PointCloudFile/PC_CuvaAnalyse.h"
#include "include/PointCloudFile/PolyFitCurve.h"

int main()
{
	PolyCurveFitmoothTest();
	return 0;
}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
