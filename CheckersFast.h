#pragma once
#include <vector>
#include "ISerialize.h"
#include "net_math.h"

namespace chkfast
{
	using vdb = std::vector<double>;

	template<typename activ>
	class layer :
		public  vdb,
		public ISerialize
	{

	public:

		// Inherited via ISerialize
		void Serialize(std::ofstream&) override
		{}
		void Serialize(std::ifstream&) override
		{}
	};

	class net
		:public ISerialize
	{
		vdb m_Input;
		std::vector<layer<math::relu_activ>> m_SharedTrunk;
		layer<math::iden_activ> m_branchPolicy;
		struct VALBRA { layer<math::relu_activ> hidden; layer<math::tanh_activ> out; } m_branchValue;
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
}