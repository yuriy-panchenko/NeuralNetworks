#include "pch.h"
#include <cmath>
#include <algorithm>
#include "Turbulance.h"

namespace turbo
{
	void net::init(std::vector<int> const& topology)
	{
		assert(topology.size() > 1);

#define OLD_STYLE
#ifdef OLD_STYLE
		input.resize(topology.front());
		layers.reserve(topology.size() - 1);

		cvd* pInp{ &input };

		for (size_t i = 1; i < topology.size(); ++i)
		{
			layers.emplace_back(topology[i]);

			for (size_t u = 0; u < topology[i]; u++)
				layers.back().add(neuron{ *pInp });

			pInp = &layers.back().out();
		}
#else
		input.resize(topology.front());
		layers.reserve(topology.size() - 1);

		cvd* pInp{ &input };

		for (int i = 1; i < (int)topology.size(); ++i)
		{
			if (i == topology.size() - 1)
				layers.emplace_back(topology[i]);
			else layers.emplace_back(topology[i] + i - 1);

			for (size_t u = 0; u < topology[i]; u++)
				layers.back().add(neuron{ *pInp });

			pInp = &layers.back().out();
		}

		/*for (size_t i = 0; i < layers.size() - 1; ++i)
			for (size_t u = i; u < layers.size(); ++u)
				layers[i].add(neuron{ layers[u].out() });*/

		for (int i = 1; i < (int)layers.size() - 1; ++i)
			for (int u = i - 1; u > -1; --u)
				layers[i].add(neuron{ layers[u].out() });
#endif // 1
	}

	std::vector<double> const& net::think(cvd& data)
	{
		assert(input.size() == data.size());
		assert(!layers.empty());

		input = data;

		for (size_t i = 0; i < layers.size(); i++)
			layers[i].think();

		return layers.back().out();
	}

	double net::learn(cvd& real)
	{
		assert(real.size() == out().size());
		err = .0;
		auto itOut{ out().begin() };

		for (auto val : real)
			err += sq(val - *itOut++);

		layers.back().learn(real);
		auto itPrev{ layers.rbegin() };

		for (auto it{ std::next(itPrev) }; it != layers.rend(); itPrev = it++)
			it->learn(*itPrev);

		for (auto& l : layers)
			l.adjust();

		return err;
	}

	double net::error() const
	{
		return err;
	}

	std::vector<size_t> net::topology() const
	{
		return std::vector<size_t>();
	}

	void net::Serialize(std::istream&) {}

	void net::Serialize(std::ostream&) {}

	std::unique_ptr<INetwork> net::create()
	{
		return std::make_unique<net>();
	}

	std::unique_ptr<INetwork> net::copy()
	{
		//return std::make_unique<net>(*this);
		return std::make_unique<net>();
	}

	layer::layer(int neuron_count)
		:output(neuron_count, .0)
	{
		cells.reserve(neuron_count);
	}

	void layer::think()
	{
		auto itOut{ output.begin() };

		for (auto& n : cells)
		{
			*itOut = n.think();
			assert(!std::isnan(*itOut) && !std::isinf(*itOut));
			++itOut;
		}
	}

	void layer::learn(cvd& real)
	{
		for (size_t i = 0; i < cells.size(); ++i)
			cells[i].set_delta((output[i] - real[i]) / 2. * der(output[i]));
	}

	void layer::learn(layer const& prev)
	{
		for (size_t i = 0; i < cells.size(); ++i)
			cells[i].set_delta(prev.weighted_error(i) * der(output[i]));
	}

	double layer::weighted_error(size_t index)const
	{
		auto err{ .0 };

		for (auto& c : cells)
			err += c.weighted_error(index);

		return err;
	}

	void layer::adjust()
	{
		for (auto& c : cells)
			c.adjust();
	}

	void layer::add(neuron&& cell)
	{
		assert(cells.size() < output.size());
		cells.push_back(std::move(cell));
	}

	neuron::neuron(cvd& inp)
		:bias{ rnd() }
	{
		auto const dev{ std::sqrt(2. / inp.size()) };
		bias *= dev;
		acsons.reserve(inp.size());

		for (auto& val : inp)
			acsons.emplace_back(val, dev);
	}

	double neuron::think() const
	{
		auto ret{ bias };

		for (auto& a : acsons)
			ret += a.think();

		return act(ret);
	}

	void neuron::adjust()
	{
		auto const dErr{ learning_rate * delta };
		bias -= dErr;

		for (auto& a : acsons)
			a.adjust(dErr);
	}

	void softmax(vd& z)
	{
		auto max_val{ *std::max_element(z.begin(), z.end()) };

		auto sum{ .0 };

		for (auto& v : z)
		{
			v = std::exp(v - max_val); // stability trick
			sum += v;
		}

		for (auto& v : z)
			v /= sum;
	}
}