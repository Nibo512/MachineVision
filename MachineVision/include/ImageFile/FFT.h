#pragma once
#include "../BaseOprFile/OpenCV_Utils.h"

//��ʾͼ��Ƶ��ͼ
void ImgF_DisplayFreqImg(Mat& fftImg, Mat& freqImg);

//���ٸ���Ҷ�任
void ImgF_FFT(Mat& srcImg, Mat& complexImg);

//���ٸ���Ҷ��任
void ImgF_InvFFT(Mat& fftImg, Mat& invFFTImg);

//�˲����ԳƸ�ֵ
void IngF_SymmetricAssignment(Mat& filter);

//����ĵ�ͨ�˲���
void ImgF_GetIdealFilter(Mat &filter, int imgW, int imgH, double radius, int passMode);

//��˹��ͨ�˲���
void ImgF_GetGaussianFilter(Mat &filter, int imgW, int imgH, double radius, int passMode);

//��״�˲���
void ImgF_GetBandFilter(Mat &filter, int imgW, int imgH, double lr, double hr, int passMode);

//�����ֶ�˹¼����
void ImgF_GetBLPFFilter(Mat &filter, int imgW, int imgH, double radius, int n, int passMode);

//̬ͬ�˲���
void ImgF_GetHomoFilter(Mat &filter, int imgW, int imgH, double radius, double L, double H, double c);

//��ȡƵ���˲���
void ImgF_GetFilter(Mat& filter, int imgW, int imgH, double lr, double hr, int passMode, IMGF_MODE filterMode);

void FFTTest();