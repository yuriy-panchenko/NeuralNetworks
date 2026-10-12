#include "pch.h"
#include "CheckersFast.h"

namespace chkf
{
	net::net()
		:net{ topology{} }
	{}

	net::net(topology const& s)
		: m_Learns{}
		, m_Adjusts{}
	{
		reshape(s);
	}

	net::out_pair net::think(vdb const& inp)
	{
		auto* pOut{ &inp };

		for (auto& l : m_sharedTrunk)
			pOut = &l.think(*pOut);

		auto const pSharedOut{ pOut };

		for (auto& l : m_policyHidden)
			pOut = &l.think(*pOut);
		m_policyOut.think(*pOut);

		pOut = pSharedOut;
		for (auto& l : m_valueHidden)
			pOut = &l.think(*pOut);
		m_valueOut.think(*pOut);

		return get_out();
	}

	void net::learn(vdb const& inp, vdb const& dL_policy, double real_value)
	{
		auto itShared{ m_sharedTrunk.crbegin() };

		vdb up_value, trunk_grad;
		{
			vdb const dL_value{ (value() - real_value) * 2. };
			vdb const* pOut{ &(m_valueHidden.empty() ? *itShared : m_valueHidden.back()) };
			auto err{ m_valueOut.learn(*pOut, dL_value) };

			for (auto iter{ m_valueHidden.rbegin() }; iter != m_valueHidden.rend(); ++iter)
			{
				if (std::next(iter) == m_valueHidden.rend())	//	first layer
					pOut = &*itShared;
				else pOut = &*std::next(iter);

				err = iter->learn(*pOut, err);
			}

			std::swap(up_value, err);
		}
		{
			vdb const* pOut{ &(m_policyHidden.empty() ? *itShared : m_policyHidden.back()) };
			auto err{ m_policyOut.learn(*pOut, dL_policy) };

			for (auto iter{ m_policyHidden.rbegin() }; iter != m_policyHidden.rend(); ++iter)
			{
				if (std::next(iter) == m_policyHidden.rend())	//	first layer
					pOut = &*itShared;
				else pOut = &*std::next(iter);

				err = iter->learn(*pOut, err);
			}

			std::swap(trunk_grad, err);
		}

		auto iter{ up_value.begin() };

		for (auto& val : trunk_grad)
			val += *iter++;

		for (auto it{ m_sharedTrunk.rbegin() }; it != m_sharedTrunk.rend(); ++it)
		{
			itShared = std::next(it);
			if (itShared == m_sharedTrunk.crend())
				trunk_grad = it->learn(inp, trunk_grad);
			else
				trunk_grad = it->learn(*itShared, trunk_grad);
		}

		++m_Learns;
	}

	void net::adjust(double lcoo)
	{
		for (auto& l : m_sharedTrunk)
			l.adjust(lcoo);

		for (auto& l : m_policyHidden)
			l.adjust(lcoo);
		m_policyOut.adjust(lcoo);

		for (auto& l : m_valueHidden)
			l.adjust(lcoo);
		m_valueOut.adjust(lcoo);

		++m_Adjusts;
	}

	std::ofstream& operator<<(std::ofstream& s, std::vector<uint64_t> const& v)
	{
		unsigned __int64 u64{ v.size() };
		s.write(reinterpret_cast<char const*>(&u64), sizeof(u64));
		s.write(reinterpret_cast<char const*>(v.data()), u64 * sizeof(uint64_t));
		return s;
	}

	std::ifstream& operator>>(std::ifstream& s, std::vector<uint64_t>& v)
	{
		unsigned __int64 u64;
		s.read(reinterpret_cast<char*>(&u64), sizeof(u64));
		v.resize(u64);
		s.read(reinterpret_cast<char*>(v.data()), u64 * sizeof(uint64_t));
		return s;
	}

	std::ofstream& operator<<(std::ofstream& s, net::topology const& t)
	{
		s << t.trunk << t.policyHidden << t.valueHidden;
		return s;
	}

	std::ifstream& operator>>(std::ifstream& s, net::topology& t)
	{
		s >> t.trunk >> t.policyHidden >> t.valueHidden;
		return s;
	}

	void net::Serialize(std::ofstream& s)
	{
		s.write(reinterpret_cast<char const*>(&magic_number), sizeof magic_number);
		s.write(reinterpret_cast<char const*>(&m_Learns), sizeof m_Learns);
		s.write(reinterpret_cast<char const*>(&m_Adjusts), sizeof m_Adjusts);

		s << get_topology();

		for (auto& l : m_sharedTrunk)
			s << l;
		for (auto& l : m_policyHidden)
			s << l;
		for (auto& l : m_valueHidden)
			s << l;

		s << m_policyOut << m_valueOut;
	}

	void net::Serialize(std::ifstream& s)
	{
		unsigned __int32 u32;
		s.read(reinterpret_cast<char*>(&u32), sizeof u32);
		if (u32 != magic_number)
			throw std::exception{ "Wrong Magic number" };

		s.read(reinterpret_cast<char*>(&m_Learns), sizeof m_Learns);
		s.read(reinterpret_cast<char*>(&m_Adjusts), sizeof m_Adjusts);

		topology sh;
		s >> sh;
		reshape(sh);
		for (auto& l : m_sharedTrunk)
			s >> l;
		for (auto& l : m_policyHidden)
			s >> l;
		for (auto& l : m_valueHidden)
			s >> l;

		s >> m_policyOut >> m_valueOut;
	}

	/*void net::shock()
	{
		for (auto& l : m_SharedTrunk)
			l.shock();

		m_branchPolicy.shock();
		m_branchValue.hidden.shock();
		m_branchValue.tail.shock();
	}*/

	net::topology net::get_topology() const
	{
		topology ret;
		ret.trunk.clear();
		ret.policyHidden.clear();
		ret.valueHidden.clear();

		for (auto& l : m_sharedTrunk)
			ret.trunk.push_back(l.size());

		for (auto& l : m_policyHidden)
			ret.policyHidden.push_back(l.size());

		for (auto& l : m_valueHidden)
			ret.valueHidden.push_back(l.size());

		return ret;
	}

	void net::reshape(topology const& s)
	{
		m_sharedTrunk.clear();
		m_policyHidden.clear();
		m_valueHidden.clear();

		auto in{ board_encoding_size };

		for (auto w : s.trunk)
		{
			m_sharedTrunk.emplace_back(in, w);
			in = w;
		}

		size_t const trunkOut{ in };                        // heads start from here

		in = trunkOut;

		for (auto w : s.policyHidden)
		{
			m_policyHidden.emplace_back(in, w);
			in = w;
		}

		m_policyOut = { in, move_space_count };

		in = trunkOut;

		for (auto w : s.valueHidden)
		{
			m_valueHidden.emplace_back(in, w);
			in = w;
		}

		m_valueOut = { in, 1 };
	}
}
