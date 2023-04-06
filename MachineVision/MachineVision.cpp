// MachineVision.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include "include/BaseOprFile/utils.h"
#include "include/BaseOprFile/OpenCV_Utils.h"

int main()
{
	cv::Mat image = cv::imread("D:/data/缺陷图片/image/1.png", 1);
	cv::Mat imggeT = image.clone();
	PC_XYZ srcPC, dstPC;

	string camFileName = "D:/data/trackPC.ply";
	pcl::io::loadPLYFile(camFileName, srcPC);

	//Eigen::Matrix4f transMatPC = Eigen::Matrix4f::Identity();
	//Eigen::Matrix4f transMatPCT = transMatPC.transpose();
	//cout << transMatPC << endl << endl;
	//cout << transMatPCT << endl << endl;

	pcl::visualization::PCLVisualizer viewer;
	viewer.addCoordinateSystem(10);

	//for (int i = 0; i < srcPC.size(); ++i)
	//{
	//	string text = to_string(i + 1);
	//	/*	viewer.addText(text, srcPC[0].x, srcPC[0].y);*/
	//	viewer.addText3D(text, srcPC[i], 5.0, 1.0, 1.0, 1.0, text);
	//}

	pcl::visualization::PointCloudColorHandlerCustom<P_XYZ> red(srcPC.makeShared(), 255, 0, 0); //设置点云颜色
	viewer.addPointCloud(srcPC.makeShared(), red, "srcPC");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "srcPC");

	//pcl::visualization::PointCloudColorHandlerCustom<P_XYZ> blue(m_WorldPts.makeShared(), 0, 255, 0); //设置点云颜色
	//viewer.addPointCloud(m_WorldPts.makeShared(), blue, "m_WorldPts");
	//viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 8, "m_WorldPts");

	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}
	return(0);

    std::cout << "Hello World!\n";
}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
