#include "pch.h"
#include "neuron.h"

namespace nnet
{
	void neuron::init(size_t acsons_count)
	{
		m_Bias = /*.0;	*/ acson::rand();
		m_dBias = m_Error = .0;
		m_Acsons.resize(acsons_count);
	}

	double neuron::think(cvd& inp) const
	{
		auto ret{ m_Bias };
		auto iter{ inp.begin() };

		for (auto& a : m_Acsons)
			ret += a.think(*iter++);

		//ret = log10(ret);
		//ret /= m_Acsons.size()+1;
		ret /= sqrt(double(m_Acsons.size() + 1));

		return act(ret);
	}

	void neuron::backpropagate(cvd::const_iterator itInp, double err)
	{
		m_dBias *= cooficient_inertial;
		m_dBias += cooficient_learning * err;
		m_Error = err;

		for (auto& a : m_Acsons)
			a.backpropagate(*itInp++ * err);

		//adjust();
	}

	double neuron::weighted_error(size_t index) const
	{
		return m_Error * m_Acsons[index].weigth();
	}

	double neuron::error() const
	{
		return m_Error;
	}

	void neuron::adjust()
	{
		m_Bias += m_dBias;
		//m_Bias = std::clamp(m_Bias,-1.,+1.);

		for (auto& a : m_Acsons)
			a.adjust();
	}

	std::ostream& operator<<(std::ostream& os, neuron const& n)
	{
		auto count{ (uint32_t)n.m_Acsons.size() };
		os.write((char const*)&n.m_Bias, sizeof(double));
		os.write((char const*)&count, sizeof uint32_t);
		for (auto& a : n.m_Acsons)
			os.write((char const*)&a.m_W, sizeof(double));
		return os;
	}
	std::istream& operator>>(std::istream& is, neuron& n)
	{
		uint32_t count;
		is.read((char*)&n.m_Bias, sizeof(double));
		is.read((char*)&count, sizeof uint32_t);
		n.m_Acsons.resize(count);
		n.m_dBias = .0;
		for (auto& a : n.m_Acsons)
		{
			is.read((char*)&a.m_W, sizeof(double));
			a.m_dW = .0;
		}
		return is;
	}

	void neuron::operator+=(neuron const& oth)
	{
		auto itAcson{ oth.m_Acsons.begin() };

		m_Bias += oth.m_Bias;

		for (auto& a : m_Acsons)
			a.m_W += (itAcson++)->m_W;
	}

	void neuron::operator/=(double db)
	{
		m_Bias /= db;

		for (auto& a : m_Acsons)
			a.m_W /= db;
	}

	bool neuron::operator==(neuron const& oth) const
	{
		if (m_Bias != oth.m_Bias)
			return false;

		if (m_Acsons.size() != oth.m_Acsons.size())
			return false;

		for (size_t i = 0; i < m_Acsons.size(); ++i)
			if (m_Acsons[i] != oth.m_Acsons[i])
				return false;

		return true;
	}
}
