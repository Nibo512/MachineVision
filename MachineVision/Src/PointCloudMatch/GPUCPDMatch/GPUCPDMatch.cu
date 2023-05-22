//#include "../../../include/PointCloudMatch/GPUCPDMatch/GPUCPDMatch.cuh"
//#include <cusolverDn.h>
//#include <fstream>
//#include <iostream>
//
////G矩阵的逆=====================================================================
//void CalGMatInvKernel(float* pA, int M, float* eigenVal)
//{
//	cusolverDnHandle_t handle;
//	cusolverDnCreate(&handle);
//	int info_gpu = 0;
//
//	float* d_A = NULL; cudaMalloc((void**)&d_A, sizeof(float) * M * M);
//	float* d_W = NULL; cudaMalloc((void**)&d_W, sizeof(float) * M);
//	int* devInfo = NULL; cudaMalloc((void**)&devInfo, sizeof(int));
//	cudaMemcpy(d_A, pA, sizeof(float) * M * M, cudaMemcpyHostToDevice);//数据从主机端传至设备端
//
//	float* d_work = NULL;
//	int lwork = 0;
//	cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; 
//	cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;  //由于是对称的，这里采用上三角
//
//	//计算GPU空间
//	cusolverDnSsyevd_bufferSize(handle, jobz, uplo, M, d_A, M, d_W, &lwork);//计算evd计算所需存储空间,保存到lwork中
//	cudaMalloc((void**)&d_work, sizeof(float) * lwork);
//
//	//特征分解
//	cusolverDnSsyevd(handle, jobz, uplo, M, d_A, M, d_W, d_work, lwork, devInfo);
//	cudaDeviceSynchronize();
//
//	//数据传回主机
//	cudaMemcpy(pA, d_A, sizeof(float) * M * M, cudaMemcpyDeviceToHost);
//	cudaMemcpy(eigenVal, d_W, sizeof(float) * M, cudaMemcpyDeviceToHost);
//	cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
//	cusolverDnDestroy(handle);
//}
////==============================================================================

////求矩阵的逆====================================================================
//void GPUCalMatInv(float* pA, int M, float* pC)
//{
//	cublasHandle_t handle;
//	cublasCreate_v2(&handle);
//
//	int* INFO;
//	int* P;
//	cudaMalloc((void**)&INFO, sizeof(int));
//	cudaMalloc((void**)&P, sizeof(int));
//
//	float* d_A;
//	float* d_C;
//	cudaMalloc((void**)&d_A, M * M * sizeof(float));
//	cudaMalloc((void**)&d_C, M * M * sizeof(float));
//	cudaMemcpy(d_A, pA, M * M * sizeof(float), cudaMemcpyHostToDevice);
//
//	float** A = (float**)malloc(sizeof(float*));
//	float** A_d;
//	cudaMalloc((void**)&A_d, sizeof(float*));
//	A[0] = d_A;
//	cudaMemcpy(A_d, A, sizeof(float*), cudaMemcpyHostToDevice);
//
//	//LU分解
//	cublasSgetrfBatched(handle, M, A_d, M, P, INFO, 1);
//	int* INFOh = new int[1];
//	cudaMemcpy(INFOh, INFO, sizeof(int), cudaMemcpyDeviceToHost);
//
//	float* C[1];
//	float** C_d;
//	cudaMalloc((void**)&C_d, sizeof(float*));
//	C[0] = d_C;
//	cudaMemcpy(C_d, C, sizeof(float*), cudaMemcpyHostToDevice);
//	cublasSgetriBatched(handle, M, A_d, M, P, C_d, M, INFO, 1);
//
//	cudaMemcpy(INFOh, INFO, sizeof(int), cudaMemcpyDeviceToHost);
//	cudaMemcpy(pC, d_C, M * M * sizeof(float), cudaMemcpyDeviceToHost);
//	cudaFree(A_d); free(A);
//	cudaFree(C_d);
//	cudaFree(INFO); 
//	cudaFree(d_A);
//	cudaFree(d_C);
//	cublasDestroy_v2(handle); 
//	if (INFOh != nullptr)
//	{
//		delete[] INFOh;
//		INFOh = nullptr;
//	}
//}
////==============================================================================
//
////矩阵相乘======================================================================
//void GPUCalMatMul(float* pA, float* pB, float *pC, int m, int n, int k)
//{
//	cublasHandle_t handle;
//	cublasCreate_v2(&handle);
//	cublasOperation_t tranpose = CUBLAS_OP_N;
//
//	float* d_A;
//	float* d_B;
//	float* d_C;
//	cudaMalloc((void**)&d_A, m * k * sizeof(float));
//	cudaMalloc((void**)&d_B, k * n * sizeof(float));
//	cudaMalloc((void**)&d_C, m * n * sizeof(float));
//	cudaMemcpy(d_A, pA, m * k * sizeof(float), cudaMemcpyHostToDevice);
//	cudaMemcpy(d_B, pB, k * n * sizeof(float), cudaMemcpyHostToDevice);
//
//	const float beta[1] = { 0.0f };
//	const float alpha[1] = { 1.0f };
//	cublasSgemm_v2(handle, tranpose, tranpose, m, n, k, alpha, d_A, m, d_B, k, beta, d_C, m);
//	cudaMemcpy(pC, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);
//
//	cublasDestroy_v2(handle);
//	cudaFree(d_A);
//	cudaFree(d_B);
//	cudaFree(d_C);
//}
////==============================================================================
