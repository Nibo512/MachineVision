#include "../BaseOprFile/utils.h"

class BarCodeDectet
{
public:
	BarCodeDectet()
	{}

	//��ʼ��ͼ��
	void ImgInit(const Mat &img);

	//ͼ��Ԥ����
	void ImgPreProcess();

	//�����ݶ�һ����
	float ComputeGradCoh_(float* const pInterImgData, int x, int y, int w_size);

	//�����ݶȷ���һ����
	void ComputeGradCoh(int w_size);

	//ͼ��ʴ
	void ImageErode();

	//��������
	void RegionGrowing(int w_size);

	//�Ǽ���ֵ����
	void BoxesNMS(float scoreThreshold);

	//�����붨λ
	void Location(const Mat &image);

private:
	int m_ImgW;
	int m_ImgH;
	Mat m_ResizeImg;
	Mat m_InterImg_xx;
	Mat m_InterImg_yy;
	Mat m_InterImg_xy;
	Mat m_InterEdges;
	Mat m_CohMat;
	Mat m_AngMat;
	Mat m_EdgesNumMat;

public:
	float m_ResizeCoff;
	vector<RotatedRect> m_LocationBoxes;
	vector<float> m_BboxScores;
	vector<int> m_BoxId;
};

//ͼ��Ԥ����
void ImgPreProcess(Mat &srcImg, Mat & dstImg);

void BarCodeTest();