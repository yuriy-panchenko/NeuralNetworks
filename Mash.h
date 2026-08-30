#pragma once
#include "INetwork.h"
#include <cstdlib>

namespace mash
{
	using vd = typename INetwork::vd;
	using cvd = typename INetwork::cvd;

	constexpr auto max_acson_length{ 3 };
	constexpr auto learning_rate{ .02 };

	inline double rnd() { return 2. * rand() / RAND_MAX - 1.; }
	inline double dev(size_t count) { return std::sqrt(2. / count); }
	inline double sq(double x) { return x * x; }
	inline double der(double x) { return 1. - sq(x); }

	//struct position { size_t x, y, z; };

	class neuron
	{
		bool fired{ false };
		double bias, dOut, err;
		vd ws;
		std::vector<neuron const*> ins;
		std::vector<neuron const*> outs;

	public:
		void init();
		void init(int);
		void connect(std::vector<neuron>&);
		void connect(neuron*);
		double out()const { return dOut; }
		double think();
		void think(cvd& data);
		bool is_waiting()const;
		bool ready()const;
		void set_error(double);
		bool is_eager()const;
		void learn();
		double weighted_error(neuron*)const;
		void change_weights();
		void change_weights(cvd&);
	};

	class encoder
	{
		vd plane[3];	//0-original grayscale, 1-edges, 2-blur

	public:
		void init(int i);
		size_t size()const { return plane[0].size(); }
		void forward(cvd& v);
		cvd& get_plane(size_t i)const { return plane[i]; }
	};

	class decoder
	{
		vd result;
		std::vector<neuron> ins;

	public:
		void init(int i) { result.resize(i, .0); ins.resize(i, {}); }
		auto& out()const { return result; }
		void connect(neuron* p);
		void think();
		double learn(cvd&);
		void correct_initial();
		void change_weights();
		size_t size()const { return ins.size(); }
	};

	//	general-purpose neural graph engine with spatial constraints
	class net
		:public INetwork
	{
		using Stride = std::vector<neuron>;
		using Matrix = std::vector<Stride>;
		using Cube = std::vector<Matrix>;

		encoder input;
		decoder output;
		size_t iSide;
		Cube cube;
		double err;

	public:
		// Inherited via INetwork
		void init(std::vector<int> const& topology) override;
		cvd& think(cvd&) override;
		double learn(vd const&) override;
		double error() const override;
		std::vector<size_t> topology() const override;
		std::unique_ptr<INetwork> create()override;
		std::unique_ptr<INetwork> copy()override;

	protected:
		void Serialize(std::ostream&) override;
		void Serialize(std::istream&) override;

	private:
		void wire_cube(int x, int y, int z);
		std::vector<neuron*> get_eager();
	};
}