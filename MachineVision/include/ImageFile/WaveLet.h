#pragma once
#include "../BaseOprFile/OpenCV_Utils.h"

class WaveLetTransformer 
{
public:
	WaveLetTransformer(const string &name, uint level);

	void SetLevel(uint level) { m_Level = level; }

	uint GetLevel() { return m_Level; }

	cv::Mat GetWaveLetL() { return m_LowFilter; }

	cv::Mat GetWaveLetH() { return m_HighFilter; }

	//����С���ֽ��˲���
	void SetFilter(const string& name);

	//����С���ع��˲���
	void Set_I_Filter(const string& name);

	//�зֽ�
	void R_Decompose(const cv::Mat& src, cv::Mat& dst_R_L, cv::Mat& dst_R_H);

	//�зֽ�
	void C_Decompose(cv::Mat& dst_R_L, cv::Mat& dst_R_H, cv::Mat& CMat1,
		cv::Mat& CMat2, cv::Mat& CMat3, cv::Mat& CMat4);

	//С�����ؽ�
	void R_Recontrcution(cv::Mat& src_C_L, cv::Mat& src_C_H, cv::Mat& dst_R, int r, int c);

	//С�����ؽ�
	void C_Reconstruction(cv::Mat& src, cv::Mat& dst_C_L, cv::Mat& dst_C_H, int r, int c);

	//��ȡż����
	void GetOddR(const cv::Mat& srcImg, cv::Mat& oddRImg);

	//��ȡż����
	void GetOddC(const cv::Mat& srcImg, cv::Mat& oddCImg);

	//�з����ֵ
	void InterC(const cv::Mat& srcImg, cv::Mat& oddCImg);

	//�з����ֵ
	void InterR(const cv::Mat& srcImg, cv::Mat& oddCImg);

	//С���ֽ�
	void WaveletDT(const cv::Mat& srcImg);

	//С���ؽ�
	void IWaveletDT(cv::Mat& outMatImg);

private:
	cv::Mat m_LowFilter;
	cv::Mat m_HighFilter;
	cv::Mat m_LowFilter_T;
	cv::Mat m_HighFilter_T;

	cv::Mat m_Low_I_Filter;
	cv::Mat m_High_I_Filter;
	cv::Mat m_Low_I_Filter_T;
	cv::Mat m_High_I_Filter_T;

	cv::Mat m_Decompose;
	uint m_Level;
	string m_Name;
};

void WaveLetTest();