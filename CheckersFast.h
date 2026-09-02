#pragma once
#include <vector>
#include <random>
#include "ISerialize.h"
#include "net_math.h"

namespace chkf
{
	using vdb = std::vector<double>;
	using cdb = std::vector<double const>;

	//	row is one neuron with elements as weights
	class matrix
	{
		size_t m_Cx, m_Cy;
		vdb m_Data;
	public:
		matrix(size_t acson_count, size_t neuron_count);
		matrix(size_t acsons, size_t neurons, double initial_value);
		//matrix(matrix const&);
		matrix(matrix&&) = default;

		//double operator()(size_t x, size_t y)const;
		//double& operator()(int x, int y);
		double const* row(size_t)const;
		double* row(size_t);
		double* ptr();
		double const* ptr()const;
		double const* ptr_end()const { return ptr() + m_Cx * m_Cy; }
		double* ptr_end() { return ptr() + m_Cx * m_Cy; }
		void randomize(double scale);
		double product(size_t iRow, vdb const& inp)const;
		size_t acsons()const { return m_Cx; }
		size_t neurons()const { return m_Cy; }
		void reset(double val) { std::fill(m_Data.begin(), m_Data.end(), val); }
	private:
		static double rnd()
		{
			static std::mt19937 rng{ std::random_device{}() };
			return std::normal_distribution<double>{.0, 1.}(rng);
		}
	};

	template<typename activ>
	class layer :
		public  vdb,
		public ISerialize
	{
		vdb m_Bias, m_dBias;
		matrix m_Weights, m_dWs;

	public:
		layer() = default;
		layer(layer const&) = default;
		layer(layer&&) = default;
		layer(size_t acsons, size_t neurons);
		// Inherited via ISerialize
		void Serialize(std::ofstream& s) override
		{
			for (size_t i = 0; i < m_Bias.size(); i++)
			{
				s.write(reinterpret_cast<char const*>(&m_Bias[i]), sizeof(double));
				s.write(reinterpret_cast<char const*>(m_Weights.row(i)), m_Weights.acsons() * sizeof(double));
			}
		}
		void Serialize(std::ifstream& s) override
		{
			for (size_t i = 0; i < m_Bias.size(); i++)
			{
				s.read(reinterpret_cast<char*>(&m_Bias[i]), sizeof(double));
				s.read(reinterpret_cast<char*>(m_Weights.row(i)), m_Weights.acsons() * sizeof(double));
			}
		}

		vdb const& think(vdb const&);
		vdb learn(vdb const& inp, vdb const&);
		void adjust(double lcoo);
	private:
	};

	class net
		:public ISerialize
	{
		//vdb m_Input;
		std::vector<layer<math::relu_activ>> m_SharedTrunk;
		layer<math::iden_activ> m_branchPolicy;
		struct { layer<math::relu_activ> hidden; layer<math::tanh_activ> tail; } m_branchValue;
		size_t m_Learns, m_Adjusts;

	public:
		using out_pair = std::pair<vdb, double>;

	public:
		net();
		void init() {}
		out_pair think(vdb const& inp);
		vdb const& policy_logits() const { return m_branchPolicy; }
		double value() const { return m_branchValue.tail.front(); }
		out_pair get_out()const { return{ m_branchPolicy, value() }; }
		void learn(vdb const& inp, vdb const& dL_policy, double real_value);
		void adjust(double lcoo);
		size_t get_learns()const { return m_Learns; }
		size_t get_adjusts()const { return m_Adjusts; }

	protected:
		// Inherited via ISerialize
		void Serialize(std::ofstream&) override;
		void Serialize(std::ifstream&) override;
	};

	////////////////////////////////////////////////////////////////////////////////////////////////////////
	template<typename activ>
	layer<activ>::layer(size_t acsons, size_t neurons)
		:vdb(neurons)
		, m_Weights{ acsons, neurons }
		, m_dWs{ acsons, neurons, .0 }
		, m_Bias(neurons, .0)
		, m_dBias(neurons, .0)
	{
		m_Weights.randomize(std::sqrt(2. / acsons));
	}

	template<typename activ>
	vdb const& layer<activ>::think(vdb const& inp)
	{
		for (size_t i = 0; i < size(); ++i)
			(*this)[i] = activ::f(m_Bias[i] + m_Weights.product(i, inp));

		return *this;
	}

	template<typename activ>
	vdb layer<activ>::learn(vdb const& inp, vdb const& dL_output)
	{
		vdb upstream(inp.size(), .0);   // size = this layer's input dim

		for (size_t iCell = 0; iCell < size(); ++iCell)
		{

			//m_Cells[j].learn(dL_doutput[j] * activ_func::df((*this)[j]), upstream);
			auto pWs{ m_Weights.row(iCell) };
			auto pdWs{ m_dWs.row(iCell) };
			auto const delta{ dL_output[iCell] * activ::df((*this)[iCell]) };

			for (size_t iAx = 0; iAx < inp.size(); ++iAx)
			{
				//upstream[i] += (*this)[i].learn(delta);
				*(pdWs + iAx) += delta * inp[iAx];
				upstream[iAx] = delta * *(pWs + iAx);
			}

			m_dBias[iCell] += delta;
		}

		return upstream;
	}

	template<typename activ>
	void layer<activ>::adjust(double lcoo)
	{
		std::transform(
			m_Weights.ptr(),
			m_Weights.ptr_end(),
			m_dWs.ptr(),
			m_Weights.ptr(),
			[lcoo](double W, double dW) {return W - lcoo * dW; });
		m_dWs.reset(.0);

		std::transform(
			m_Bias.begin(),
			m_Bias.end(),
			m_dBias.begin(),
			m_Bias.begin(),
			[lcoo](double b, double db) {return b - lcoo * db; });
		std::fill(m_dBias.begin(), m_dBias.end(), .0);


		//m_Bias -= dErr * m_dBias;
	//	m_dBias = .0;
	}
}