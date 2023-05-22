#include "../../../include/PointCloudMatch/CPDMatch/GPUCPD.h"
#include <fstream>
#include <iostream>

//G矩阵的逆=====================================================================
void GPUSVDCalMatEigenVal(cusolverDnHandle_t& cusolverHandle, float* pD_A, int M, float* pD_C)
{
	//float* d_W = NULL; cudaMalloc((void**)&d_W, sizeof(float) * M);
	int* devInfo = NULL; cudaMalloc((void**)&devInfo, sizeof(int));

	float* d_work = NULL;
	int lwork = 0;
	cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
	cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;  //由于是对称的，这里采用上三角

	//计算GPU空间
	cusolverDnSsyevd_bufferSize(cusolverHandle, jobz, uplo, M, pD_A, M, pD_C, &lwork);//计算evd计算所需存储空间,保存到lwork中
	cudaMalloc((void**)&d_work, sizeof(float) * lwork);

	//特征分解
	cusolverDnSsyevd(cusolverHandle, jobz, uplo, M, pD_A, M, pD_C, d_work, lwork, devInfo);
	cudaDeviceSynchronize();

	cudaFree(devInfo);
	cudaFree(d_work);
}
//==============================================================================

//求矩阵的逆====================================================================
void GPUCalMatInv(cublasHandle_t &cublasHandle, float* pD_A, int M, float* pD_C)
{
	int* INFO;
	int* P;
	cudaMalloc((void**)&INFO, sizeof(int));
	cudaMalloc((void**)&P, sizeof(int));

	float** A = new float*[1];
	float** A_d;
	cudaMalloc((void**)&A_d, sizeof(float*));
	A[0] = pD_A;
	cudaMemcpy(A_d, A, sizeof(float*), cudaMemcpyHostToDevice);

	//LU分解
	cublasSgetrfBatched(cublasHandle, M, A_d, M, P, INFO, 1);
	//int* INFOh = new int[1];
	//cudaMemcpy(INFOh, INFO, sizeof(int), cudaMemcpyDeviceToHost);

	float* C[1];
	float** C_d;
	cudaMalloc((void**)&C_d, sizeof(float*));
	C[0] = pD_C;
	cudaMemcpy(C_d, C, sizeof(float*), cudaMemcpyHostToDevice);
	cublasSgetriBatched(cublasHandle, M, A_d, M, P, C_d, M, INFO, 1);

	//cudaMemcpy(INFOh, INFO, sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(A_d);
	cudaFree(C_d);
	cudaFree(INFO);
	delete[] A;
	A = nullptr;
}
//==============================================================================

//矩阵的SVD分解=================================================================
void GPUCalMatSVD(cusolverDnHandle_t& cusolverHandle, float* pA, int M, int N, Eigen::MatrixXf &invMat)
{
	Eigen::MatrixXf Source(M, N);
	cudaMemcpy(Source.data(), pA, M * N * sizeof(float), cudaMemcpyDeviceToHost);

	float* pU;
	float* pVH;
	float* pS;
	cudaMalloc((void**)&pU, M * M * sizeof(float));
	cudaMalloc((void**)&pVH, N * N * sizeof(float));
	cudaMalloc((void**)&pS, N * sizeof(float));
	int lwork = 0;
	cusolverDnSgesvd_bufferSize(cusolverHandle, M, N, &lwork);

	//float* d_rwork  = nullptr;
	//cudaMalloc(&d_rwork, sizeof(float) * (M - 1));
	int* devInfo;
	cudaMalloc((void**)&devInfo, sizeof(int));
	float* d_work;
	cudaMalloc((void**)&d_work, sizeof(float) * lwork);

	signed char jobu = 'A';
	signed char jobvt = 'A';

	cusolverDnSgesvd(cusolverHandle, jobu, jobvt, M, N, pA, M, pS, pU, M, pVH, N, d_work, lwork, nullptr, devInfo);
	
	float* pS_H = new float[N];
	Eigen::MatrixXf UMat(M, M);
	Eigen::MatrixXf VMatH(N, N);
	Eigen::MatrixXf SMat = Eigen::MatrixXf::Zero(M, N);
	cudaMemcpy(UMat.data(), pU, M * M * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(VMatH.data(), pVH, N * N * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(pS_H, pS, N * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(pU); cudaFree(pVH); cudaFree(pS);
	/*cudaFree(d_rwork);*/ cudaFree(devInfo); cudaFree(d_work);
	//delete[] d_rwork;
	Eigen::MatrixXf diagS = Eigen::MatrixXf::Zero(M,N);

	//Eigen::MatrixXf IMat = VMatH * VMatH.transpose();
	for (int i = 0; i < M; ++i)
	{
		diagS(i, i) = 1.0f / pS_H[i];
	}
	invMat = (VMatH.transpose() * diagS * UMat.transpose())/* *  Source*/;

	//fstream file("D:/file.csv", ios::app);
	//for (int i = 0; i < M; ++i)
	//{
	//	for (int j = 0; j < M; ++j)
	//	{
	//		file << diff(i, j) << ",";
	//	}
	//	file << endl;
	//}

	//file.close();
	delete[] pS_H;
}
//==============================================================================