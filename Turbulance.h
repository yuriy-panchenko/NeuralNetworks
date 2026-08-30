#pragma once
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include "INetwork.h"

namespace turbo
{
	using vd = typename INetwork::vd;
	using cvd = typename INetwork::cvd;

	constexpr double learning_rate{ .01 };
	//constexpr double inertial_rate{ .1 };

	inline double rnd() { return 2. * rand() / RAND_MAX - 1.; }
	inline double sq(double x) { return x * x; }
	inline double act(double x) { return tanh(x); }
	inline double der(double x) { return 1. - sq(x); }
	void softmax(vd& z);

	class acson
	{
		double w;
		double const& input;

	public:
		acson(double const& inp, double dev) :input{ inp }, w{ rnd() * dev } {}
		double think()const { return w * input; }
		double weight()const { return w; }
		void adjust(double err) { w -= err * input; }
	};

	class neuron
	{
		double bias, delta;
		std::vector<acson> acsons;

	public:
		neuron(cvd& inp);
		double think()const;
		void set_delta(double db) { delta = db; }// std::clamp(db, -1., 1.);

		double weighted_error(size_t index)const { return index < acsons.size() ? acsons[index].weight() * delta : .0; }
		void adjust();
	};

	class layer
	{
		std::vector<neuron> cells;
		std::vector<double> output;

	public:
		layer(int neuron_count);

		std::vector<double> const& out()const { return output; }
		void think();
		void learn(cvd&);
		void learn(layer const&);
		void adjust();
		void add(neuron&&);

	private:
		double weighted_error(size_t)const;
	};

	class net
		:public INetwork
	{
		vd input;
		std::vector<layer> layers;
		double err;

	public:
		net() = default;
		net(net const&) {}
		net(net&&) {}
		~net() noexcept override {} // <-- FIX: add noexcept to match base class

		net const& operator=(net const&) { return *this; }
		void operator=(net&&) {}

		cvd& out()const { return layers.back().out(); }

	public:
		// Inherited via INetwork
		void init(std::vector<int> const& topology) override;
		cvd& think(cvd&) override;
		double learn(cvd&) override;
		double error() const override;
		std::vector<size_t> topology() const override;

		std::unique_ptr<INetwork> create()override;
		std::unique_ptr<INetwork> copy()override;

	private:
		void Serialize(std::ostream&) override;
		void Serialize(std::istream&) override;
	};
}