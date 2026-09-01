#pragma once

namespace math
{
	struct relu_activ
	{
		static double f(double x) { return x > .0 ? x : .0; }
		static double df(double y) { return y > .0 ? 1. : .0; }
	};

	struct tanh_activ
	{
		static double f(double x) { return ::tanh(x); }
		static double df(double y) { return 1. - y * y; }
	};

	struct iden_activ
	{
		static double f(double x) { return x; }
		static double df(double y) { return 1.; }
	};
}