#include "../../include/PointCloudFile/ReadSTLFile.h"
#include "../../include/BaseOprFile/MathOpr.h"

//读取stl文件===========================================================
bool ReadSTLFile(const string &filename, PC_XYZ &pointcloud)
{
	if (filename.empty())
		return false;
	ifstream in(filename, ios::in);
	if (!in)
		return false;
	string headStr;
	std::getline(in, headStr, ' ');
	in.close();
	if (headStr.empty())
		return false;
	if (headStr[0] == 's')
		ReadSTLASCII(filename);
	else
		ReadSTLBinary(filename, pointcloud);
}
//======================================================================

//读取ASCII文件=========================================================
bool ReadSTLASCII(const string &filename)
{
	PC_XYZ pointcloud;
	int i = 0, j = 0, cnt = 0, pCnt = 4;
	char a[200];
	char str[100];
	double x = 0, y = 0, z = 0;
	ifstream in;
	in.open(filename, ios::in);
	if (!in)
		return false;
	do {
		i = 0; cnt = 0;
		in.getline(a, 100, '\0');
		while (a[i] != '\0')
		{
			if (!islower((int)a[i]) && !isupper((int)a[i]) && a[i] != ' ')
				break;
			cnt++; i++;
		}
		while (a[cnt] != '\0')
		{
			str[j] = a[cnt];
			cnt++; i++;
		}
		str[j] = '\0';
		j = 0;
		if (sscanf(str, "%1f%1f%1f", &x, &y, &z) == 3)
		{
			pointcloud.push_back({ (float)x, (float)y, (float)z });
		}
		pCnt++;
	} while (!in.eof());
	return true;
}
//======================================================================

//读取二值化文件========================================================
bool ReadSTLBinary(const string &filename, PC_XYZ &pointcloud)
{
	char str[80];
	ifstream in;
	in.open(filename, ios::in | ios::binary);
	if (!in)
		return false;
	in.read(str, 80);
	int unTriangles;
	in.read((char*)&unTriangles, sizeof(int));
	if (unTriangles == 0)
	{
		return false;
	}
	for (int i = 0; i < unTriangles; ++i)
	{
		float coorXYZ[12];
		in.read((char*)coorXYZ, 12 * sizeof(float));
		PC_XYZ pt;
		for (int j = 1; j < 4; ++j)
		{
			pt.push_back({ coorXYZ[3*j], coorXYZ[3 * j + 1], coorXYZ[3 * j + 2] });
		}
		PC_XYZ dstPC;
		FillTrangle(pt, dstPC);

		pointcloud += dstPC;
		in.read((char*)coorXYZ, 2);
	}
	in.close();
	return true;
}
//======================================================================

//填充三角形============================================================
void FillTrangle(PC_XYZ &pt, PC_XYZ &dstPC)
{
	P_XYZ norm1, norm2;
	PC_PPVec(pt[0], pt[1], norm1);
	PC_PPVec(pt[0], pt[2], norm2);
	float dist1 = PC_PPDist(pt[0], pt[1]);
	float dist2 = PC_PPDist(pt[0], pt[2]);
	if (dist1 < dist2)
	{
		std::swap(dist1, dist2);		
		std::swap(norm1, norm2);
	}
	float ratio = dist2 / dist1;
	for (float step = 0; step <= dist1; step += 0.8)
	{
		float off_x1 = norm1.x * step, off_y1 = norm1.y * step, off_z1 = norm1.z * step;
		float off_x2 = norm2.x * step * ratio, off_y2 = norm2.y * step * ratio, off_z2 = norm2.z * step * ratio;
		P_XYZ pt_s(pt[0].x + off_x1, pt[0].y + off_y1, pt[0].z + off_z1);
		P_XYZ pt_e(pt[0].x + off_x2, pt[0].y + off_y2, pt[0].z + off_z2);
		P_XYZ norm_;
		PC_PPVec(pt_s, pt_e, norm_);
		float dist = PC_PPDist(pt_s, pt_e);
		for (float step_ = 0; step_ <= dist; step_ += 0.8)
		{
			dstPC.push_back({ pt_s.x + step_ * norm_.x, pt_s.y + step_ * norm_.y, pt_s.z + step_ * norm_.z });
		}
	}
}
//======================================================================
