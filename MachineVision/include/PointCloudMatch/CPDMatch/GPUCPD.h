#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cusolverDn.h>
#include "../../BaseOprFile/utils.h"

//�����ı���̬�뱾��ֵ
void GPUSVDCalMatEigenVal(cusolverDnHandle_t& cusolverHandle, float* pA, int M, float* eigenVal);

//��������
void GPUCalMatInv(cublasHandle_t& cublasHandle, float* pA, int M, float* pC);

//�����SVD�ֽ�
void GPUCalMatSVD(cusolverDnHandle_t& cusolverHandle, float* pA, int M, int N, Eigen::MatrixXf& invMat);