#include "pch.h"
#include "CheckersFast.h"

namespace chkf
{
	matrix::matrix(size_t acson_count, size_t neuron_count)
		:m_Cx{ acson_count }
		, m_Cy{ neuron_count }
		, m_Data(acson_count* neuron_count)
	{}

	inline double const* matrix::row(size_t row_index) const
	{
		return ptr() + row_index * m_Cx;
	}

	inline double* matrix::ptr()
	{
		return m_Data.data();
	}

	inline double const* matrix::ptr()const
	{
		return m_Data.data();
	}

	net::net()
		:m_Input{ 128 }
		, m_branchPolicy{ 128, 896 }
		, m_branchValue{ {128, 64}, {64, 1} }
		, m_Learns{}
		, m_Adjusts{}
	{
		m_SharedTrunk.reserve(3);
		m_SharedTrunk.emplace_back(128, 256);
		m_SharedTrunk.emplace_back(256, 256);
		m_SharedTrunk.emplace_back(256, 128);
	}
}
