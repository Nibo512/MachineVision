#pragma once
#include "../BaseOprFile/OpenCV_Utils.h"

/*ͼ��ĻҶȹ�����*/
// ö�ٻҶȹ�������ķ���
enum GLCM_ORIT
{
	GLCM_HORIZATION = 0,		// ˮƽ
	GLCM_VERTICAL = 1,			// ��ֱ
	GLCM_ANGLE45 = 2,			// 45�Ƚ�
	GLCM_ANGLE135 = 3			// 135�Ƚ�
};

//����Ҷȹ�����
void Img_ComputeGLCM(Mat& srcImg, Mat& MatGLCM, int grayLevel, GLCM_ORIT angle);

// ����ˮƽ�Ҷȹ�������
void Img_GLCM_0(Mat& srcImg, Mat& MatGLCM);

// ���㴹ֱ�Ҷȹ�������
void Img_GLCM_90(Mat& srcImg, Mat &MatGLCM);

// ���� 45 �ȻҶȹ�������
void Img_GLCM_45(Mat& srcImg, Mat& MatGLCM);

// ���� 135 �ȻҶȹ�������
void Img_GLCM_135(Mat& srcImg, Mat& MatGLCM);

//������Ƶ��
void cal_total_number(Mat &MatGLCM, int &number);

//����Ҷȹ����ص�����/һ����
void Img_GLCMEnergy(Mat& MatGLCM, double& energy);

//��ûҶȹ����ص�����ͬ����
void Img_GLCMHomogeneity(Mat& MatGLCM, double& homogeneity);

//��û�������صĶԱȶ�
void Img_GLCMContrast(Mat& MatGLCM, double& contrast);

//��ûҶȹ����������ض�
void Img_GLCMCorrelation(Mat& MatGLCM, double& correlation);

//��ûҶȹ����ص���
void Img_GLCMEntropy(Mat& MatGLCM, double& entropy);

//���ux��uy
void get_ux_uy(Mat &MatGLCM, float &ux, float &uy);

//���sx��sy
void get_sx_sy(Mat &MatGLCM, float &ux, float &uy, float &sx, float &sy);
