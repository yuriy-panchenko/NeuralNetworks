#pragma once
#include "INetwork.h"

namespace mash2d
{
	using vd = INetwork::vd;
	using cvd = INetwork::cvd;

	constexpr auto learning_coo{ .1 };
	static constexpr int acson_length{ 3 };

	inline double sq(double x) { return x * x; }
	inline double rnd() { return double(rand()) / RAND_MAX * 2. - 1.; }
	inline double der(double x) { return 1. - sq(x); }

	class neuron;

	class acson
	{
		neuron* m_pInput;
		double m_W;

	public:
		acson(neuron& inp) :m_pInput{ &inp } {}
		bool same(neuron const* p)const { return p == m_pInput; }
		double think()const;
		void set_weight(double w) { m_W = w; }
		void back_propagate(double err)const;
		void change(double db);
	};

	class neuron
	{
		std::vector<acson> acsons;
		bool fired;
		double bias, out, err;

	public:
		neuron();

		void connect(neuron&);
		void set_out(double);
		void think();
		double get_out()const noexcept { assert(fired); return out; }
		double get_out(int)const noexcept { return out; }
		//	init_biases_and_weights
		void ibaw();
		bool is_fired()const noexcept { return fired; }
		void set_real_error(double);
		void accum_error(double db) { err += db; }
		void propagate_error();
		void adjust();
		void reset_error() { err = .0; }
	};

	class layer
	{
		std::vector<neuron> cells;

	public:
		layer(int);
		size_t side()const;
		neuron& get_cell(size_t);
		void set_input(cvd&);
		auto& get_cells()const { return cells; }
		auto& get_cells() { return cells; }
		void init_biases_and_weights();
	};

	class net
		:public INetwork
	{
	public:
		// Inherited via INetwork
		void init(std::vector<int> const& topology) override;
		cvd& think(cvd&) override;
		double learn(cvd&) override;
		double error() const override;
		std::vector<size_t> topology() const override;

		std::unique_ptr<INetwork> create()override;
		std::unique_ptr<INetwork> copy()override;

	protected:
		void Serialize(std::ostream&) override;
		void Serialize(std::istream&) override;

	private:
		void assert_all_neurons_fired(bool)const;

		std::vector<layer> m_Layers;
		//std::vector<neuron*> m_exeLine;

		vd m_Output;
		double m_Error;
	};
}
