#pragma once
#include <algorithm>
namespace nnet
{
	using vd = std::vector<double>;
	using cvd = std::vector<double> const;
	constexpr double
		cooficient_inertial{ .25 },
		cooficient_learning{ .03 };

	inline double sq(double val)
	{
		return val * val;
	}

	inline double derr(double out)
	{
		return 1. - sq(out);
	}

	inline double act(double val)
	{
		return tanh(val);
	}

	class acson
	{
	public:
		double m_W, m_dW;

	public:
		acson() :m_W{ rand() }, m_dW{ .0 } {}
		double think(double val)const { return val * m_W; }
		static double rand() /*{ return ::rand() / RAND_MAX; }*/ { return  2. * ::rand() / RAND_MAX - 1.; }
		void backpropagate(double err)
		{
			m_dW *= cooficient_inertial;
			m_dW += cooficient_learning * err;
		}
		double weigth()const { return m_W; }
		void adjust() { m_W += m_dW; }
		bool operator==(acson const& oth)const { return m_W == oth.m_W; }
	};
}