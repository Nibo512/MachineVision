#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cusolverDn.h>
#include "../../BaseOprFile/utils.h"

//求矩阵的本征态与本征值
void GPUSVDCalMatEigenVal(cusolverDnHandle_t& cusolverHandle, float* pA, int M, float* eigenVal);

//求矩阵的逆
void GPUCalMatInv(cublasHandle_t& cublasHandle, float* pA, int M, float* pC);

//矩阵的SVD分解
void GPUCalMatSVD(cusolverDnHandle_t& cusolverHandle, float* pA, int M, int N, Eigen::MatrixXf& invMat);