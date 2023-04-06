#include "../../include/ImageFile/ImageFilter.h"
#include "omp.h"

//�����˲�=============================================================================
void Img_GuidFilter(Mat &srcImg, Mat &guidImg, Mat &dstImg, int size, float eps)
{
	if (!dstImg.empty())
		dstImg.release();
	cv::Size imgSize = srcImg.size();
	dstImg = Mat(imgSize, srcImg.type(), cv::Scalar(0));

	Mat SrcImg_32f(imgSize, CV_32F, cv::Scalar(0));
	Mat GuidImg_32f(imgSize, CV_32F, cv::Scalar(0));

	srcImg.convertTo(SrcImg_32f, CV_32F);
	guidImg.convertTo(GuidImg_32f, CV_32F);

	cv::Size winSize(size, size);
	//���� I*I��I*P
	Mat img_SG = SrcImg_32f.mul(GuidImg_32f);
	Mat img_GG = GuidImg_32f.mul(GuidImg_32f);

	//�����ֵ
	Mat mean_S, mean_G, mean_SG, mean_GG;
	cv::boxFilter(srcImg, mean_S, CV_32F, winSize);
	cv::boxFilter(guidImg, mean_G, CV_32F, winSize);
	cv::boxFilter(img_SG, mean_SG, CV_32F, winSize);
	cv::boxFilter(img_GG, mean_GG, CV_32F, winSize);

	//���� IP ��Э��������Լ� I �ķ������
	Mat var_IP = mean_SG - mean_G.mul(mean_S);
	Mat var_II = mean_GG - mean_G.mul(mean_G) + eps * eps;

	//���� a��b;
	Mat a, b;
	cv::divide(var_IP, var_II, a);
	b = mean_S - a.mul(mean_G);
	//���� a��b �ľ�ֵ
	Mat mean_a, mean_b;
	cv::boxFilter(a, mean_a, CV_32F, winSize);
	cv::boxFilter(b, mean_b, CV_32F, winSize);

	uchar* pDstImg = dstImg.ptr<uchar>();
	float* pFGuidImg = GuidImg_32f.ptr<float>();
	float* pMean_a = mean_a.ptr<float>();
	float* pMean_b = mean_b.ptr<float>();
	int step = imgSize.width * srcImg.channels();
	for (int y = 0; y < imgSize.height; ++y, pDstImg += step,
		pMean_a += step, pMean_b += step, pFGuidImg += step)
	{
		for (int x = 0; x < step; ++x)
		{
			float value = pMean_a[x] * pFGuidImg[x] + pMean_b[x];
			if (value > 0 && value < 256)
				pDstImg[x] = value;
			if (value > 255)
				pDstImg[x] = 255;
		}
	}
}
//=====================================================================================

//����ӦCanny�˲�======================================================================
void Img_AdaptiveCannyFilter(Mat &srcImg, Mat &dstImg, int size, double sigma)
{
	cv::Scalar midVal = cv::mean(srcImg);
	double minVal = midVal[0] * (1 - sigma);
	double maxVal = midVal[0] * (1 + sigma);
	cv::Canny(srcImg, dstImg, minVal, maxVal, size);
}
//=====================================================================================

//Ƶ�����˲�===========================================================================
void ImgF_FreqFilter(Mat &srcImg, Mat &dstImg, double lr, double hr, int passMode, IMGF_MODE filterMode)
{
	int imgH = srcImg.rows, imgW = srcImg.cols;
	int imgH_ = getOptimalDFTSize(imgH);
	int imgW_ = getOptimalDFTSize(imgW);
	Mat filter;
	ImgF_GetFilter(filter, imgW_, imgH_, lr, hr, passMode, filterMode);

	Mat srcImg_32F;
	srcImg.convertTo(srcImg_32F, CV_32F);

	int channels = srcImg_32F.channels();
	vector<Mat> idftImg(srcImg_32F.channels());
	Mat* pSplitImg = new Mat[channels];
	Mat* pMergeImg = new Mat[channels];
	cv::split(srcImg_32F, pSplitImg);
#pragma omp parallel for
	for (int i = 0; i < channels; ++i)
	{
		Mat fftImg;
		ImgF_FFT(pSplitImg[i], fftImg);
		Mat fftFilterImg = fftImg.mul(filter);

		Mat invFFTImg;
		ImgF_InvFFT(fftFilterImg, invFFTImg);
		Mat roi = invFFTImg(Rect(0, 0, imgW, imgH));
		double minVal, maxVal;
		minMaxLoc(roi, &minVal, &maxVal, NULL, NULL);
		roi = (roi - minVal) / (maxVal - minVal) * 255.0;
		roi.convertTo(pMergeImg[i], CV_8UC1);
	}
	cv::merge(pMergeImg, channels, dstImg);
	delete[] pSplitImg;
	delete[] pMergeImg;
}
//=====================================================================================

//̩ͬ�˲�=============================================================================
void ImgF_HomoFilter(Mat &srcImg, Mat &dstImg, double radius, double L, double H, double c)
{
	int imgH = srcImg.rows, imgW = srcImg.cols;
	
	Mat filter;
	int M = getOptimalDFTSize(imgH);
	int N = getOptimalDFTSize(imgW);
	ImgF_GetHomoFilter(filter, N, M, radius, L, H, c);

	Mat srcImg_32F;
	srcImg.convertTo(srcImg_32F, CV_32F);
	int channels = srcImg_32F.channels();
	vector<Mat> idftImg(srcImg_32F.channels());
	Mat* pSplitImg = new Mat[channels];
	Mat* pMergeImg = new Mat[channels];
	cv::split(srcImg_32F, pSplitImg);
#pragma omp parallel for
	for (int i = 0; i < channels; ++i)
	{
		cv::log((pSplitImg[i] + 1), pSplitImg[i]);
		Mat fftImg;
		ImgF_FFT(pSplitImg[i], fftImg);
		Mat fftFilterImg = fftImg.mul(filter);
		Mat invFFTImg;
		ImgF_InvFFT(fftFilterImg, invFFTImg);
		Mat roi = invFFTImg(Rect(0, 0, imgW, imgH));
		cv::normalize(roi, roi, 0, 1, NORM_MINMAX);
		cv::exp(roi, roi);
		double minVal, maxVal;
		minMaxLoc(roi, &minVal, &maxVal, NULL, NULL);
		roi = (roi - minVal) / (maxVal - minVal) * 255.0;
		roi.convertTo(pMergeImg[i], CV_8UC1);
	}
	cv::merge(pMergeImg, channels, dstImg);
	delete[] pSplitImg;
	delete[] pMergeImg;
}
//=====================================================================================

//��������ƽ��=========================================================================
void Img_AnisotropicFilter(Mat &srcImg, Mat &dstImg, double lamda, double step_t, int iter_k)
{
	dstImg = srcImg.clone();
	int row = srcImg.rows;
	int col = srcImg.cols;
	int channel = srcImg.channels();
	int step = col * srcImg.channels();

	double lamda_ = 1.0 / lamda * 1.0 / lamda;
	for (int i = 0; i < iter_k; ++i)
	{
#pragma omp parallel for
		for (int y = 1; y < row-1; ++y)
		{
			uchar* pDstUp = dstImg.ptr<uchar>(y-1);
			uchar* pDst = dstImg.ptr<uchar>(y);
			uchar* pDstDown = dstImg.ptr<uchar>(y+1);
			for (int x = 1; x < step - channel; x += channel)
			{
				for (int c = 0; c < channel; ++c)
				{
					int x_ = x + c;
					double dstVal = (double)pDst[x_];
					double dxUp = ((double)pDstUp[x_] - dstVal);
					double dxDown = ((double)pDstDown[x_] - dstVal);
					double dyLeft = ((double)pDst[x_ - channel] - dstVal);
					double dyRight = ((double)pDst[x_ + channel] - dstVal);

					double dxUpz_2 = dxUp * std::exp(-dxUp * dxUp * lamda_);
					double dxDown_2 = dxDown * std::exp(-dxDown * dxDown * lamda_);
					double dyLeft_2 = dyLeft * std::exp(-dyLeft * dyLeft * lamda_);
					double dyRight_2 = dyRight * std::exp(-dyRight * dyRight * lamda_);

					double value = dstVal + step_t * (dxUpz_2 + dxDown_2 + dyLeft_2 + dyRight_2);
					if (value > 0 && value < 256)
						pDst[x_] = value;
				}
			}
		}
	}
}
//=====================================================================================

//��˹�˲�=============================================================================
void Img_GaussFilter(Mat &srcImg, Mat &dstImg, int h, int w)
{
	Mat kernel_x(1, 2 * w + 1, CV_64FC1), kernel_y(2 * h + 1, 1, CV_64FC1);
	double sigma_x = double(w) / 3.0;
	double sigma_x1 = 1.0 / (sigma_x * sigma_x);
	double sigma_x_ = 1.0 / (sigma_x1 * CV_PI);
	double sum_kx = 0.0;
	for (int i = 0; i < 2 * w + 1; ++i)
	{
		double x_ = i - w;
		kernel_x.ptr<double>(0)[i] = sigma_x_ * std::exp(-x_ * x_ * sigma_x1);
		sum_kx += kernel_x.ptr<double>(0)[i];
	}
	kernel_x /= sum_kx;
	double sigma_y = double(h) / 3.0;
	double sigma_y1 = 1.0 / (sigma_y * sigma_y);
	double sigma_y_ = 1.0 / (sigma_y1 * CV_PI);
	double sum_ky = 0.0;
	for (int i = 0; i < 2 * h + 1; ++i)
	{
		double y_ = i - w;
		kernel_y.ptr<double>(0)[i] = sigma_y_ * std::exp(-y_ * y_ * sigma_y1);
		sum_ky += kernel_y.ptr<double>(0)[i];
	}
	kernel_y /= sum_ky;

	Mat filter_x;
	cv::filter2D(srcImg, filter_x, CV_8UC3, kernel_x);
	cv::filter2D(filter_x, dstImg, CV_8UC3, kernel_y);
}
//=====================================================================================

void FilterTest()
{
	string imgPath = "D:/image/image_6.png";
	Mat srcImg = imread(imgPath, 0);
	cv::medianBlur(srcImg, srcImg, 5);

	cv::Mat sobelImg_x, sobelImg_y;
	cv::Sobel(srcImg, sobelImg_x, CV_32FC1, 1, 0, 9);
	cv::Sobel(srcImg, sobelImg_y, CV_32FC1, 0, 1, 9);
	cv::Mat apmImg(cv::Size(srcImg.cols, srcImg.rows), CV_32FC1, cv::Scalar::all(0));
	float maxVal = 0.0f, minVal = 1e8f;
	for (int y = 0; y < srcImg.rows; ++y)
	{
		float *pDx = sobelImg_x.ptr<float>(y);
		float *pDy = sobelImg_y.ptr<float>(y);
		float *pAmp = apmImg.ptr<float>(y);
		for (int x = 0; x < srcImg.cols; ++x)
		{
			float dx = pDx[x]/* > 0 ? pDx[x] : 0*/;
			float dy = pDy[x] /*> 0 ? pDy[x] : 0*/;
			pAmp[x] = std::abs(dx) + std::abs(dy)/*std::sqrt(dx * dx + dy * dy)*/;
			maxVal = maxVal > pAmp[x] ? maxVal : pAmp[x];
			minVal = minVal < pAmp[x] ? minVal : pAmp[x];
		}
	}
	float scale = 255.0f / (maxVal - minVal);
	cv::Mat bitAmpImg(cv::Size(srcImg.cols, srcImg.rows), CV_8UC1, cv::Scalar::all(0));
	for (int y = 0; y < srcImg.rows; ++y)
	{
		uchar *pBitImg = bitAmpImg.ptr<uchar>(y);
		float *pAmp = apmImg.ptr<float>(y);
		for (int x = 0; x < srcImg.cols; ++x)
		{
			pBitImg[x] = (pAmp[x] - minVal) * scale;
		}
	}
	cv::Mat cannyImg;
	cv::Canny(bitAmpImg, cannyImg, 500, 800, 5);
	//equalizeHist(bitAmpImg, bitAmpImg);
	Mat dstImg;
	ImgF_FreqFilter(bitAmpImg, dstImg, 50, 200, 1, IMGF_MODE::IMGF_GAUSSIAN);

	cv::Mat binImg(cv::Size(srcImg.cols, srcImg.rows), CV_8UC1, cv::Scalar::all(0));
	cv::threshold(bitAmpImg, binImg, 10, 255, cv::THRESH_BINARY);

	cv::Mat t = binImg.clone();
}