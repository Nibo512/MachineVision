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

/*约定：
	srcPC：原点云
    dstPC：处理后的输出点云
*/

//体素滤波---下采样
void PC_VoxelGrid(PC_XYZ& srcPC, PC_XYZ& dstPC, float leafSize);

//直通滤波----保留指定方向上指定区域的点云
void PC_PassFilter(PC_XYZ &srcPC, PC_XYZ &dstPC, const string mode, double minVal, double maxVal);

/*导向滤波*/
void PC_GuideFilter(PC_XYZ& srcPC, PC_XYZ& dstPC, int radius, double lamda);

//降采样
void PC_DownSample(PC_XYZ& srcPC, PC_XYZ& dstPC, float size, int mode);

//测试程序
void PC_FitlerTest();