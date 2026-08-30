#pragma once
#include "INetwork.h"
#include "layer.h"

namespace nnet
{
	class net
		:public INetwork
	{
	public:
		void init(std::vector<int> const& topology);
		cvd& think(cvd&);
		double learn(vd const&);
		double error()const;
		std::vector<size_t> topology()const;
		void operator+=(net const&);
		void operator/=(double);
		bool operator==(net const&)const;

	private:
		vd m_Input;
		std::vector<layer> m_Layers;
		double m_Error;

		// Inherited via INetwork
		void Serialize(std::ostream&) override;
		void Serialize(std::istream&) override;

		// Inherited via INetwork
		std::unique_ptr<INetwork> create() override;
		std::unique_ptr<INetwork> copy() override;
	};
}