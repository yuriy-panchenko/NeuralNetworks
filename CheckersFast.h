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
	using RELU_layer = uni::layer<math::relu_activ<Type>>;
	using IdentityLayer = uni::layer<math::iden_activ<Type>>;
	using TanhLayer = uni::layer<math::tanh_activ<Type>>;

	class net
		:public ISerialize
	{
		std::vector<RELU_layer>
			m_sharedTrunk,
			m_policyHidden,
			m_valueHidden;

		IdentityLayer m_policyOut;
		TanhLayer m_valueOut;

		size_t m_Learns, m_Adjusts;

	public:
		static constexpr size_t
			board_encoding_size{ 128ull },                                   // board encoding
			move_space_count{ 896ull };
		using out_pair = std::pair<vdb, double>;
		struct topology
		{
			std::vector<uint64_t>
				trunk{ 256, 256, 128 },
				policyHidden{ },      // empty = same as today
				valueHidden{ 64 };
		};

	public:
		net();
		net(topology const&);
		void init() {}
		out_pair think(vdb const& inp);
		vdb const& policy_logits() const { return m_policyOut; }
		double value() const { return m_valueOut.front(); }
		out_pair get_out()const { return{ m_policyOut, value() }; }
		void learn(vdb const& inp, vdb const& dL_policy, double real_value);
		void adjust(double lcoo);
		size_t get_learns()const { return m_Learns; }
		size_t get_adjusts()const { return m_Adjusts; }
		//void shock();
		topology get_topology()const;
		void reshape(topology const&);

	protected:
		// Inherited via ISerialize
		void Serialize(std::ofstream&) override;
		void Serialize(std::ifstream&) override;
	};
}