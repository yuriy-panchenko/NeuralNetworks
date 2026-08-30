#pragma once
#include "acson.h"
namespace nnet
{
	class neuron
	{
		std::vector<acson> m_Acsons;
		double m_Bias, m_dBias, m_Error;

	public:
		void init(size_t acsons_count);
		double think(cvd&)const;
		void backpropagate(cvd::const_iterator itInp, double err);
		double weighted_error(size_t)const;
		double error()const;
		void adjust();

		friend std::ostream& operator<<(std::ostream&,neuron const &);
		friend std::istream& operator>>(std::istream&,neuron&);

		void operator+=(neuron const&);
		void operator/=(double);
		bool operator==(neuron const&)const;
		//bool operator!=(neuron const&)const;
	};
}