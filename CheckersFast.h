#pragma once
#include <vector>
#include <random>
#include "ISerialize.h"
#include "universal.h"
#include "net_math.h"

namespace chkf
{
	using Type = double;
	using vdb = std::vector<Type>;
	using cdb = std::vector<Type const>;

	class net
		:public ISerialize
	{
		std::vector<uni::layer<math::relu_activ<Type>>> m_SharedTrunk;
		uni::layer<math::iden_activ<Type>> m_branchPolicy;
		struct { uni::layer<math::relu_activ<Type>> hidden; uni::layer<math::tanh_activ<Type>> tail; } m_branchValue;
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
		void shock();

	protected:
		// Inherited via ISerialize
		void Serialize(std::ofstream&) override;
		void Serialize(std::ifstream&) override;
	};
}