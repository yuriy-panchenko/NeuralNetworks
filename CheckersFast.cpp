#include "pch.h"
#include "CheckersFast.h"

namespace chkf
{
	net::net()
		:/*m_Input{ 128 }
		,*/ m_branchPolicy{ 128, 896 }
		, m_branchValue{ {128, 64}, {64, 1} }
		, m_Learns{}
		, m_Adjusts{}
	{
		m_SharedTrunk.reserve(3);
		m_SharedTrunk.emplace_back(128, 256);
		m_SharedTrunk.emplace_back(256, 256);
		m_SharedTrunk.emplace_back(256, 128);
	}

	net::out_pair net::think(vdb const& inp)
	{
		auto* pOut{ &inp };

		for (auto& l : m_SharedTrunk)
			pOut = &l.think(*pOut);

		m_branchPolicy.think(*pOut);

		pOut = &m_branchValue.hidden.think(*pOut);
		m_branchValue.tail.think(*pOut);

		return get_out();
	}

	void net::learn(vdb const& inp, vdb const& dL_policy, double real_value)
	{
		auto itShared{ m_SharedTrunk.crbegin() };
		auto const up_policy{ m_branchPolicy.learn(*itShared, dL_policy) };          // 128-dim
		vdb const dL_value{ (value() - real_value) * 2. };
		auto const up_hidden{ m_branchValue.tail.learn(m_branchValue.hidden,dL_value) };         // 64-dim
		auto const up_value{ m_branchValue.hidden.learn(*itShared, up_hidden) };     // 128-dim

		auto trunk_grad{ up_policy };
		auto iter{ up_value.begin() };

		for (auto& val : trunk_grad)
			val += *iter++;

		for (auto it{ m_SharedTrunk.rbegin() }; it != m_SharedTrunk.rend(); ++it)
		{
			itShared = std::next(it);
			if (itShared == m_SharedTrunk.crend())
				trunk_grad = it->learn(inp, trunk_grad);
			else
				trunk_grad = it->learn(*itShared, trunk_grad);
		}

		++m_Learns;
	}

	void net::adjust(double lcoo)
	{
		for (auto& l : m_SharedTrunk)
			l.adjust(lcoo);

		m_branchPolicy.adjust(lcoo);
		m_branchValue.hidden.adjust(lcoo);
		m_branchValue.tail.adjust(lcoo);
		++m_Adjusts;
	}

	void net::Serialize(std::ofstream& s)
	{
		s.write(reinterpret_cast<char const*>(&magic_number), sizeof magic_number);
		s.write(reinterpret_cast<char const*>(&m_Learns), sizeof m_Learns);
		s.write(reinterpret_cast<char const*>(&m_Adjusts), sizeof m_Adjusts);

		for (auto& l : m_SharedTrunk)
			s << l;
		s << m_branchPolicy << m_branchValue.hidden << m_branchValue.tail;
	}

	void net::Serialize(std::ifstream& s)
	{
		unsigned __int32 u32;
		s.read(reinterpret_cast<char*>(&u32), sizeof u32);
		if (u32 != magic_number)
			throw std::exception{ "Wrong Magic number" };

		s.read(reinterpret_cast<char*>(&m_Learns), sizeof m_Learns);
		s.read(reinterpret_cast<char*>(&m_Adjusts), sizeof m_Adjusts);

		for (auto& l : m_SharedTrunk)
			s >> l;
		s >> m_branchPolicy >> m_branchValue.hidden >> m_branchValue.tail;
	}

	void net::shock()
	{
		for (auto& l : m_SharedTrunk)
			l.shock();
	}
}
