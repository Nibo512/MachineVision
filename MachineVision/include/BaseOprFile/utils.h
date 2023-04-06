#pragma once

/*OpenCV�汾��4.5.3
  PCL�汾��1.9.1
*/

#include <iostream>
#include <string>
#include <conio.h>
#include <vector>
#include <random> 

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/kdtree/kdtree_flann.h>  
#include <pcl/search/kdtree.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/normal_3d.h>

#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_sphere.h>

#include <pcl/common/transforms.h>
#include <pcl/common/common.h>  
#include <pcl/sample_consensus/model_types.h>
#include <pcl/registration/icp.h>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#ifdef NB
#define NB_API _declspec(dllexport )
#endif

const double EPS = 1e-8;

enum NB_MODEL_FIT_METHOD {
	OLS_FIT = 0,
	HUBER_FIT = 1,
	TUKEY_FIT = 2
};

using namespace std;
using namespace pcl;
using namespace cv;

typedef pcl::PointCloud<pcl::PointXYZ> PC_XYZ;
typedef pcl::PointXYZ P_XYZ;
typedef pcl::PointCloud<pcl::PointNormal> PC_XYZN;
typedef pcl::PointCloud<pcl::Normal> PC_N;
typedef pcl::PointNormal P_XYZN;
typedef pcl::PointCloud<pcl::PointXYZI> PC_XYZI;
typedef pcl::PointXYZI P_XYZI;
typedef pcl::Normal P_N;
typedef pcl::PointIndices P_IDX;

//ֱ��
typedef struct Line2D
{
	double a;
	double b;
	double c;
}Line2D;

//Բ
typedef struct Circle2D
{
	double x;
	double y;
	double r;
}Circle2D;

//��Բ-----�������꣺(x, y)����x��нǣ�angle�������᣺(a, b)
typedef struct Ellipse2D
{
	double x;
	double y;
	double angle;
	double a;
	double b;
}Ellipse2D;

//ƽ��
typedef struct Plane3D
{
	double a;
	double b;
	double c;
	double d;
}Plane3D;

//�ռ�԰----�������꣺(x, y, z)�� �뾶��r��ԭ����ƽ�淨������(a, b, c)
typedef struct Circle3D
{
	double x;
	double y;
	double z;
	double r;
	double a;
	double b;
	double c;
}Circle3D;

//��
typedef struct Sphere3D
{
	double x;
	double y;
	double z;
	double r;
}Sphere3D;

//�ռ�ֱ��----ֱ�ߵķ���������(a, b, c)��ֱ���ϵ����꣺(x, y, z)
typedef struct Line3D
{
	double a;
	double b;
	double c;
	double x;
	double y;
	double z;
}Line3D;

class NB_Array2D
{
public:
	enum DataFlag {
		Pt2i = 0,
		Pt2f = 1,
		Pt2d = 2,
	};

public:
	template<typename _Tp> NB_Array2D(const _Tp& vec)
	{
		m_Size = vec.size();
		if (m_Size == 0)
			return;
		m_pData = (void*)vec.data();
		if (typeid(Point2i) == typeid(vec[0]))
			m_Flags = Pt2i;
		else if (typeid(Point2f) == typeid(vec[0]))
			m_Flags = Pt2f;
		else if (typeid(Point2d) == typeid(vec[0]))
			m_Flags = Pt2d;
	}
	const Point2d operator[](int n)
	{
		if (m_Flags == Pt2i)
			return (Point2d)((Point2i*)m_pData)[n];
		else if (m_Flags == Pt2f)
			return (Point2d)((Point2f*)m_pData)[n];
		else if (m_Flags == Pt2d)
			return ((Point2d*)m_pData)[n];
	}
	int size()
	{
		return m_Size;
	}

private:
	void* m_pData;
	int m_Size;
	DataFlag m_Flags;
};

class NB_Array3D
{
public:
	template<typename _Tp> NB_Array3D(const _Tp& vec)
	{
		m_Size = vec.size();
		m_vData.resize(m_Size);
		//����ط�Ū�ɴ�ָ�룬�����ǿ������ݣ�������
		for (int i = 0; i < m_Size; ++i)
		{
			m_vData[i].x = vec[i].x; m_vData[i].y = vec[i].y; m_vData[i].z = vec[i].z;
		}
	}
	Point3d& operator[](int n)
	{
		return m_vData[n];
	}
	int size()
	{
		return m_Size;
	}

private:
	vector<Point3d> m_vData;
	int m_Size;
};
