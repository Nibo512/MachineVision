#pragma once
#include "../BaseOprFile/utils.h"
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/bilateral.h> 
#include <pcl/filters/fast_bilateral.h>  
#include <pcl/filters/median_filter.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/convolution_3d.h>
#include <pcl/filters/morphological_filter.h>
#include <pcl/filters/model_outlier_removal.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/surface/mls.h>

/*Լ����
	srcPC��ԭ����
    dstPC���������������
*/

//�����˲�---�²���
void PC_VoxelGrid(PC_XYZ& srcPC, PC_XYZ& dstPC, float leafSize);

//ֱͨ�˲�----����ָ��������ָ������ĵ���
void PC_PassFilter(PC_XYZ &srcPC, PC_XYZ &dstPC, const string mode, double minVal, double maxVal);

/*�����˲�*/
void PC_GuideFilter(PC_XYZ& srcPC, PC_XYZ& dstPC, int radius, double lamda);

//������
void PC_DownSample(PC_XYZ& srcPC, PC_XYZ& dstPC, float size, int mode);

//���Գ���
void PC_FitlerTest();