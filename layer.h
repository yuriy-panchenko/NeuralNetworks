#pragma once
#include "neuron.h"
namespace nnet
{
	class layer
		:public vd
	{
		std::vector<neuron> m_Cells;

	public:
		void init(size_t acsons, size_t cells);
		void think(cvd& input);
		void set_real(cvd::const_iterator itInp, cvd& real);
		void backpropagate(cvd::const_iterator itInp, layer const& nextLayer);
		void adjust();
		size_t cell_count()const;

		friend std::ostream& operator<<(std::ostream&, layer const&);
		friend std::istream& operator>>(std::istream&, layer&);

		void operator+=(layer const&);
		void operator/=(double);
		bool operator==(layer const&)const;
		bool operator!=(layer const&)const;

	private:
		double weighted_error(size_t)const;
	};
}