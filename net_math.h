#pragma once

namespace math
{
	template<typename T>
	struct relu_activ
	{
		using type = T;
		static T f(T x) { return x > T(0) ? x : T(0); }
		static T df(T y) { return y > T(0) ? T(1) : T(0); }
	};

	template<typename T>
	struct tanh_activ
	{
		using type = T;
		static T f(T x) { return ::tanh(x); }
		static T df(T y) { return T(1) - y * y; }
	};

	template<typename T>
	struct iden_activ
	{
		using type = T;
		static T f(T x) { return x; }
		static T df(T y) { return T(1); }
	};	
}