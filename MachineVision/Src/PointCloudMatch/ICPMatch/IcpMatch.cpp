#include "../../../include/PointCloudMatch/ICPMatch/IcpMatch.h"

//я╟урвНаз╫Э╣Ц====================================================================
void ICP::FindKnnPair()
{
	int ptNum = m_SrcPC.size();
	P_XYZ* pSrc = m_SrcPC.points.data();
	m_PairIdxes.resize(0);
	m_PairIdxes.reserve(ptNum);

	for (int i = 0; i < ptNum; ++i)
	{
		vector<int> PIdx;
		vector<float> PDist;
		m_TgtKdTree.nearestKSearch(pSrc[i], 1, PIdx, PDist);
		if (PDist[0] < m_MaxPPDist)
		{
			m_PairIdxes.push_back(PairIdx(i, PIdx[0]));
		}
	}
}
//================================================================================