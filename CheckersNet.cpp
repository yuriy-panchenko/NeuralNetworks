#include "pch.h"
#include "CheckersNet.h"

namespace chk
{
	neuron::neuron(vdb const& inp)
		:m_Bias{ }
		, m_dBias{}
	{
		reserve(inp.size());
		double const scale{ sqrt(2. / inp.size()) };

		for (auto& val : inp)
			emplace_back(val, scale);
	}

	double neuron::sum() const
	{
		auto ret{ m_Bias };

		for (auto& a : *this)
			ret += a.think();

		return ret;
	}

	void neuron::learn(double const delta, vdb& upstream)
	{
		for (size_t i = 0; i < size(); ++i)
			upstream[i] += (*this)[i].learn(delta);

		m_dBias += delta;
	}

	void neuron::adjust(double dErr)
	{
		for (auto& a : *this)
			a.adjust(dErr);

		m_Bias -= dErr * m_dBias;
		m_dBias = .0;
	}

	void neuron::Serialize(std::ofstream& s)
	{
		s.write(reinterpret_cast<char const*>(&m_Bias), sizeof m_Bias);
		for (auto& a : *this)
			s << a;
	}

	void neuron::Serialize(std::ifstream& s)
	{
		s.read(reinterpret_cast<char*>(&m_Bias), sizeof m_Bias);
		for (auto& a : *this)
			s >> a;
	}

	void net::init()
	{
		m_Input.resize(128);
		m_SharedTrunk.clear();
		m_branchPolicy.clear();
		m_branchValue.hidden.clear();
		m_branchValue.out.clear();
		m_SharedTrunk.reserve(3);

		m_SharedTrunk.emplace_back(m_Input, 256);
		m_SharedTrunk.emplace_back(m_SharedTrunk.back(), 256);
		m_SharedTrunk.emplace_back(m_SharedTrunk.back(), 128);

		m_branchPolicy = { m_SharedTrunk.back(), 896 };

		m_branchValue.hidden = { m_SharedTrunk.back(), 64 };
		m_branchValue.out = { m_branchValue.hidden, 1 };

		m_Learns = m_Adjusts = 0ull;
	}

	void net::think(vdb const& inp)
	{
		assert(inp.size() == m_Input.size());
		m_Input = inp;

		for (auto& l : m_SharedTrunk)
			l.think();

		m_branchPolicy.think();

		m_branchValue.hidden.think();
		m_branchValue.out.think();
	}

	void net::learn(vdb const& dL_policy, double target_value)
	{
		auto const up_policy{ m_branchPolicy.learn(dL_policy) };          // 128-dim
		vdb const dL_value{ (value() - target_value) * 2. };
		auto const up_hidden{ m_branchValue.out.learn(dL_value) };         // 64-dim
		auto const up_value{ m_branchValue.hidden.learn(up_hidden) };     // 128-dim

		auto trunk_grad{ up_policy };
		auto iter{ up_value.begin() };

		for (auto& val : trunk_grad)
			val += *iter++;

		for (auto it{ m_SharedTrunk.rbegin() }; it != m_SharedTrunk.rend(); ++it)
			trunk_grad = it->learn(trunk_grad);

		++m_Learns;
	}

	void net::adjust(double const dErr)
	{
		for (auto& l : m_SharedTrunk)
			l.adjust(dErr);

		m_branchPolicy.adjust(dErr);
		m_branchValue.hidden.adjust(dErr);
		m_branchValue.out.adjust(dErr);
		++m_Adjusts;
	}

	void net::Serialize(std::ofstream& s)
	{
		s.write(reinterpret_cast<char const*>(&magic_number), sizeof magic_number);
		s.write(reinterpret_cast<char const*>(&m_Learns), sizeof m_Learns);
		s.write(reinterpret_cast<char const*>(&m_Adjusts), sizeof m_Adjusts);

		for (auto& l : m_SharedTrunk)
			s << l;
		s << m_branchPolicy << m_branchValue.hidden << m_branchValue.out;
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
		s >> m_branchPolicy >> m_branchValue.hidden >> m_branchValue.out;
	}
}
