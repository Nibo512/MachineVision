#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

////G矩阵的逆
//void CalGMatInvKernel(float* pA, int M, float* eigenVal);
//
////求矩阵的逆=
//void GPUCalMatInv(float* pA, int M, float* pC);
//
////矩阵相乘
//void GPUCalMatMul(float* pA, float* pB, float* pC, int m, int n, int k);