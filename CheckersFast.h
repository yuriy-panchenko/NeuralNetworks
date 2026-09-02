#pragma once
#include <vector>
#include "ISerialize.h"
#include "net_math.h"

namespace chkf
{
	using vdb = std::vector<double>;

	//	row is one neuron with elements as weights
	class matrix
	{
		size_t m_Cx, m_Cy;
		std::vector<double> m_Data;
	public:
		matrix(size_t acson_count, size_t neuron_count);
		//matrix(matrix const&);
		matrix(matrix&&) = default;

		//double operator()(size_t x, size_t y)const;
		//double& operator()(int x, int y);
		double const* row(size_t)const;
	private:
		double* ptr();
		double const* ptr()const;
	};

	template<typename activ>
	class layer :
		public  vdb,
		public ISerialize
	{
		matrix m_Weights;

	public:
		layer() = default;
		layer(layer const&) = default;
		layer(layer&&) = default;
		layer(size_t acsons, size_t neurons);
		// Inherited via ISerialize
		void Serialize(std::ofstream&) override {}
		void Serialize(std::ifstream&) override {}
	};

	class net
		:public ISerialize
	{
		vdb m_Input;
		std::vector<layer<math::relu_activ>> m_SharedTrunk;
		layer<math::iden_activ> m_branchPolicy;
		struct { layer<math::relu_activ> hidden; layer<math::tanh_activ> out; } m_branchValue;
		size_t m_Learns, m_Adjusts;

	public:
		using out_pair = std::pair<vdb, double>;

	public:
		net();
		void init();
		void think(vdb const& inp);
		vdb const& policy_logits() const { return m_branchPolicy; }
		double value() const { return m_branchValue.out.front(); }
		out_pair get_out()const { return{ policy_logits(), value() }; }
		void learn(vdb const& dL_policy, double real_value);
		void adjust(double dErr);
		size_t get_learns()const { return m_Learns; }
		size_t get_adjusts()const { return m_Adjusts; }

	protected:
		// Inherited via ISerialize
		void Serialize(std::ofstream&) override;
		void Serialize(std::ifstream&) override;
	};
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename activ>
chkf::layer<activ>::layer(size_t acsons, size_t neurons)
	:vdb(neurons)
	, m_Weights{ acsons + 1, neurons }
{}