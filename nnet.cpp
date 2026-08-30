#include "pch.h"
#include "nnet.h"

namespace nnet
{

	void net::init(std::vector<int> const& topology)
	{
		assert(topology.size());
		auto iter{ topology.begin() };
		m_Input.resize(*iter);
		m_Layers.resize(topology.size());
		m_Error = .0;
		vd const* pInput{ &m_Input };

		for (auto& l : m_Layers)
		{
			l.init(pInput->size(), *iter++);
			pInput = &l;
		}
	}

	cvd const& net::think(cvd& input)
	{
		assert(input.size() == m_Input.size());
		//std::swap(m_Input, input);
		m_Input = input;

		vd const* pInput{ &m_Input };

		for (auto& l : m_Layers)
		{
			l.think(*pInput);
			pInput = &l;
		}

		return *pInput;
	}

	double net::learn(vd const& real)
	{
		assert(!m_Layers.empty());

		auto iter{ m_Layers.rbegin() };
		{
			auto itInp{ std::next(iter) };
			cvd& inp{ itInp == m_Layers.rend() ? m_Input : *itInp };
			iter->set_real(inp.cbegin(), real);
		}

		for (iter = std::next(iter); iter != m_Layers.rend(); iter = std::next(iter))
		{
			auto itInp{ std::next(iter) };
			cvd& inp{ itInp == m_Layers.rend() ? m_Input : *itInp };
			iter->backpropagate(inp.cbegin(), *std::prev(iter));
		}

		for (auto& l : m_Layers)
			l.adjust();

		m_Error = .0;

		auto itReal{ real.begin() };
		for (auto out : m_Layers.back())
			m_Error += sq(out - *itReal++);

		m_Error /= real.size();

		return m_Error;
	}

	double net::error()const
	{
		return m_Error;
	}

	std::vector<size_t> net::topology() const
	{
		std::vector<size_t> ret;
		ret.reserve(m_Layers.size());

		for (auto& l : m_Layers)
			ret.push_back(l.cell_count());

		return ret;
	}

	void net::Serialize(std::ostream& os)
	{
		auto count{ (uint32_t)m_Layers.size() };
		os.write((char const*)&count, sizeof uint32_t);

		for (auto& l : m_Layers)
			os << l;
	}

	void net::Serialize(std::istream& is)
	{
		uint32_t count;
		is.read((char*)&count, sizeof uint32_t);
		m_Layers.resize(count);

		for (auto& l : m_Layers)
			is >> l;

		if (!m_Layers.empty())
			m_Input.resize(m_Layers.front().cell_count());
	}

	std::unique_ptr<INetwork> net::create()
	{
		return std::make_unique<net>();
	}

	std::unique_ptr<INetwork> net::copy()
	{
		return std::unique_ptr<net>(new net{ *this });
	}

	void net::operator+=(net const& oth)
	{
		assert(oth.topology() == topology());
		auto itLay{ oth.m_Layers.begin() };

		for (auto& l : m_Layers)
			l += *itLay++;
	}

	void net::operator/=(double db)
	{
		for (auto& l : m_Layers)
			l /= db;
	}

	bool net::operator==(net const& oth)const
	{
		if (topology() != oth.topology())
			return false;

		auto iter{ oth.m_Layers.begin() };

		for (auto& l : m_Layers)
			if (l != *iter++)
				return false;

		return true;
	}
}