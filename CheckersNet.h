#pragma once
#include <vector>
#include "ISerialize.h"
#include "net_math.h"
#include "CheckersFast.h"

namespace chk
{
	using Type = double;
	using vdb = std::vector<double>;

	class acson
		:public ISerialize
	{
		double m_W, m_dW;
		double const& m_Input;
	public:
		acson(double const& inp, double scale) :m_W{ rnd() * scale }, m_dW{}, m_Input{ inp } {}
		double think()const { return m_Input * m_W; }
		static double rnd() { return ::rand() * 2. / RAND_MAX - 1.; }
		double learn(double delta) { m_dW += delta * m_Input; return delta * m_W; }
		void adjust(double dErr) { m_W -= dErr * m_dW; m_dW = 0.; }

	protected:
		// Inherited via ISerialize
		void Serialize(std::ofstream& s) override { s.write(reinterpret_cast<char const*>(&m_W), sizeof m_W); }
		void Serialize(std::ifstream& s) override { s.read(reinterpret_cast<char*>(&m_W), sizeof m_W); }
	};

	class neuron
		:public std::vector<acson>
		, public ISerialize
	{
		double m_Bias, m_dBias;
	public:
		neuron(vdb const& inp);
		double sum()const;
		void learn(double delta, vdb& upstream);
		void adjust(double dErr);

	protected:
		// Inherited via ISerialize
		void Serialize(std::ofstream&) override;
		void Serialize(std::ifstream&) override;
	};

	template<typename activ_func>
	class layer
		:public vdb
		, public ISerialize
	{
		std::vector<neuron> m_Cells;
	public:
		layer() = default;
		layer(vdb const&, int);
		void think();
		vdb learn(vdb const&);
		void adjust(double dErr);
	
	protected:
		// Inherited via ISerialize
		void Serialize(std::ofstream&) override;
		void Serialize(std::ifstream&) override;
	};

	class net
		:public ISerialize
	{
		vdb m_Input;
		std::vector<layer<math::relu_activ<Type>>> m_SharedTrunk;
		layer<math::iden_activ<Type>> m_branchPolicy;
		struct VALBRA { layer<math::relu_activ<Type>> hidden; layer<math::tanh_activ<Type>> out; } m_branchValue;
		size_t m_Learns, m_Adjusts;

	public:
		void init();
		void think(vdb const& inp);
		vdb const& policy_logits() const { return m_branchPolicy; }
		double value() const { return m_branchValue.out.front(); }
		void learn(vdb const& dL_policy, double real_value);
		void adjust(double dErr);
		size_t get_learns()const { return m_Learns; }
		size_t get_adjusts()const { return m_Adjusts; }

	protected:
		// Inherited via ISerialize
		void Serialize(std::ofstream&) override;
		void Serialize(std::ifstream&) override;
	};

	/////////////////////////////////////////////////////////////////////////////////////////////////
	template<typename activ_func>
	layer<activ_func>::layer(vdb const& inp, int neuron_count)
		:vdb(neuron_count)
	{
		m_Cells.reserve(neuron_count);

		for (int i = 0; i < neuron_count; i++)
			m_Cells.emplace_back(inp);
	}

	template<typename activ_func>
	void layer<activ_func>::think()
	{
		auto iter{ begin() };

		for (auto const& n : m_Cells)
			(*iter++) = activ_func::f(n.sum());
	}

	template<typename activ_func>
	vdb layer<activ_func>::learn(vdb const& dL_doutput)
	{
		vdb upstream(m_Cells.front().size(), .0);   // size = this layer's input dim

		for (size_t j = 0; j < m_Cells.size(); ++j)
			m_Cells[j].learn(dL_doutput[j] * activ_func::df((*this)[j]), upstream);

		return upstream;
	}

	template<typename activ_func>
	void layer<activ_func>::adjust(double dErr)
	{
		for (auto& n : m_Cells)
			n.adjust(dErr);
	}

	template<typename activ_func>
	void layer<activ_func>::Serialize(std::ofstream& s)
	{
		for (auto& n : m_Cells)
			s << n;
	}

	template<typename activ_func>
	void layer<activ_func>::Serialize(std::ifstream& s)
	{
		for (auto& n : m_Cells)
			s >> n;
	}
}