#pragma once
#include "utils.h"

enum IMG_SEG {
	IMG_SEG_LIGHT = 0,
	IMG_SEG_DARK = 1,
	IMG_SEG_EQUL = 2,
	IMG_SEG_NOTEQUL = 3
};

enum IMGF_MODE {
	IMGF_IDEAL = 0,
	IMGF_GAUSSIAN = 1,
	IMGF_BAND = 2,
	IMGF_BLPF = 3
};

/*����ͼ���ֱ��ͼ*/
void Img_ComputeImgHist(Mat& srcImg, Mat& hist);

/*���ƻҶ�ֱ��ͼ*/
void Img_DrawHistImg(Mat& hist);
