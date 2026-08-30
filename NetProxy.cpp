#include "pch.h"
#include "NetProxy.h"
namespace proxy
{

	MultilayerPerceptron::MultilayerPerceptron(const std::vector<int>& layer_sizes)
		: layers(layer_sizes)
		, rng(std::random_device{}())
		, dist(0.0, 1.0)
	{
		// He initialization (good for sigmoid/tanh)
		double stddev;

		auto generate_biases = [this, &stddev](size_t count)->Vector
			{
				Vector b(count);
				std::transform(b.begin(), b.end(), b.begin(),
					[&](double) { return dist(rng) * stddev; });
				return b;
			};

		auto generate_weights = [this, &stddev](size_t rows, size_t cols)->Matrix
			{
				Matrix w(rows, Vector(cols));
				for (auto& row : w)
					std::transform(row.begin(), row.end(), row.begin(),
						[&](double) { return dist(rng) * stddev; });
				return w;
			};

		// Initialize weights and biases
		for (auto it{ layers.begin() }, itNext{ std::next(it) }; itNext != layers.end(); it = itNext++)
		{
			stddev = std::sqrt(2. / *it);
			weights.push_back(generate_weights(*itNext, *it));
			biases.push_back(generate_biases(*itNext));
		}
	}

	MultilayerPerceptron::MultilayerPerceptron(std::istream& is)
	{
		uint32_t u32;
		is.read(reinterpret_cast<char*>(&u32), sizeof u32);
		layers.resize(u32);
		is.read(reinterpret_cast<char*>(layers.data()), u32 * sizeof(int));

		is.read(reinterpret_cast<char*>(&u32), sizeof u32);
		weights.resize(u32);

		for (auto& mrx : weights)
		{
			is.read(reinterpret_cast<char*>(&u32), sizeof u32);
			mrx.resize(u32);

			if (u32)
			{
				is.read(reinterpret_cast<char*>(&u32), sizeof u32); //	cx				

				for (auto& line : mrx)
				{
					line.resize(u32);
					is.read(reinterpret_cast<char*>(line.data()), u32 * sizeof(double));
				}
			}
		}

		is.read(reinterpret_cast<char*>(&u32), sizeof u32);
		biases.resize(u32);	//	count

		for (auto& v : biases)
		{
			is.read(reinterpret_cast<char*>(&u32), sizeof u32);
			v.resize(u32);	//	count			
			is.read(reinterpret_cast<char*>(v.data()), u32 * sizeof(double));
		}
	}

	Vector MultilayerPerceptron::forward(const Vector& input)
	{
		assert(input.size() == layers[0] && "Input size mismatch");

		activations = { input };
		auto pCurr{ &activations.back() };

		auto react = [](auto itA, auto itAEnd, auto itB, double& val)
			{
				while (itA != itAEnd)
					val += *itA++ * *itB++;
			};

		auto process_layer = [react](size_t acsons, size_t neurons, Vector const& inp, Vector const& bias, Matrix const& weight)->Vector
			{
				auto z{ bias };
				auto itRow{ weight.begin() };
				// z = W * a + b
				for (auto& val : z)
				{
					react(itRow->begin(), itRow->end(), inp.begin(), val);
					++itRow;
				}

				// Apply activation
				std::transform(z.begin(), z.end(), z.begin(),
					[](double val) { return sigmoid(val); });

				return z;
			};

		for (size_t i = 0; i < weights.size(); ++i)
		{
			activations.push_back(process_layer(layers[i], layers[i + 1], *pCurr, biases[i], weights[i]));
			pCurr = &activations.back();
		}

		return *pCurr; // output
	}

	double MultilayerPerceptron::backprop(const Vector& target, double const learning_rate)
	{
		// Forward pass (recompute to get latest activations)
		int const L{ ((int)layers.size()) - 1 }; // index of output layer
		std::vector<Vector> deltas(layers.size());
		deltas[L].resize(target.size());
		std::transform(target.begin(), target.end(), activations.back().begin(), deltas[L].begin(), [](auto const a, auto const b) {return a - b; });

		double ret{ .0 };
		for (auto val : deltas[L])
			ret += val * val;

		auto gate_error = [this, &deltas](int const l)->Vector
			{
				Vector delta(layers[l], .0);
				const auto& w_next = weights[l];

				for (int j = 0; j < layers[l]; ++j)
					delta[j] = [&](int j)->double
					{
						double err{ .0 };
						for (int k = 0; k < layers[l + 1]; ++k)
							err += w_next[k][j] * deltas[l + 1][k];
						return err;
					}(j)*sigmoid_derivative(activations[l][j]);

				return delta;
			};

		// Backpropagate error
		for (int l = L - 1; l > 0; --l)
			deltas[l] = gate_error(l);

		// Update weights and biases
		for (int l = 0; l < weights.size(); ++l)
			for (int j = 0; j < weights[l].size(); ++j)
			{
				auto const dErr{ learning_rate * deltas[l + 1][j] };

				biases[l][j] += dErr;

				auto itAct{ activations[l].begin() };

				for (auto& w : weights[l][j])
					w += dErr * *itAct++;
			}

		return ret;
	}

	void MultilayerPerceptron::train(const std::vector<Vector>& inputs,
		const std::vector<Vector>& targets,
		int epochs,
		double lr) {
		assert(inputs.size() == targets.size());

		for (int epoch = 1; epoch <= epochs; ++epoch)
			for (size_t i = 0; i < inputs.size(); ++i)
			{
				forward(inputs[i]);
				backprop(targets[i], lr);
			}
	}

	std::vector<size_t> MultilayerPerceptron::topology() const
	{
		std::vector<size_t> ret;

		for (auto i : layers)
			ret.push_back(static_cast<size_t>(i));

		return ret;
	}

	void MultilayerPerceptron::save(std::ostream& os) const
	{
		auto u32{ static_cast<uint32_t>(layers.size()) };
		os.write(reinterpret_cast<char const*>(&u32), sizeof u32);
		os.write(reinterpret_cast<char const*>(layers.data()), u32 * sizeof(int));

		u32 = static_cast<uint32_t>(weights.size());	//	count
		os.write(reinterpret_cast<char const*>(&u32), sizeof u32);
		for (auto const& mrx : weights)
		{
			u32 = static_cast<uint32_t>(mrx.size());	//	cy
			os.write(reinterpret_cast<char const*>(&u32), sizeof u32);
			if (u32)
			{
				u32 = static_cast<uint32_t>(mrx.front().size());	//	cx
				os.write(reinterpret_cast<char const*>(&u32), sizeof u32);

				for (auto& line : mrx)
					os.write(reinterpret_cast<char const*>(line.data()), u32 * sizeof(double));
			}
		}

		u32 = static_cast<uint32_t>(biases.size());	//	count
		os.write(reinterpret_cast<char const*>(&u32), sizeof u32);

		for (auto& v : biases)
		{
			u32 = static_cast<uint32_t>(v.size());	//	count
			os.write(reinterpret_cast<char const*>(&u32), sizeof u32);
			os.write(reinterpret_cast<char const*>(v.data()), u32 * sizeof(double));
		}
	}

	net::net(net const& oth)
		: pNet{ oth.pNet ? std::make_unique<MultilayerPerceptron>(*oth.pNet) : nullptr }
		, m_Error{ oth.m_Error }
	{
	}

	net& net::operator=(net const& oth)
	{
		pNet = oth.pNet ? std::make_unique<MultilayerPerceptron>(*oth.pNet) : nullptr;
		m_Error = oth.m_Error;
		return *this;
	}

	void net::init(std::vector<int> const& topo)
	{
		pNet = std::make_unique<MultilayerPerceptron>(topo);
		m_Error = .0;
		m_Out.clear();
	}

	Vector const& net::think(Vector const& data)
	{
		assert(pNet);
		m_Out = pNet->forward(data);
		return m_Out;
	}

	double net::learn(Vector const& real)
	{
		assert(pNet);
		//auto real2{ real };

		//for (auto& val : real2)
		//	val = max(val, .0);

		//for (auto& val : real2)
		//	val = val > .0 ? .9 : .1;

		m_Error = pNet->backprop(real, .1);

		return m_Error;
	}

	double net::error()const
	{
		return m_Error;
	}

	std::vector<size_t> net::topology()const
	{
		if (pNet)
			return pNet->topology();
		else return {};
	}

	void net::Serialize(std::ostream& os)
	{
		if (pNet)
			pNet->save(os);
	}

	void net::Serialize(std::istream& is)
	{
		pNet = std::make_unique<MultilayerPerceptron>(is);
	}

	std::unique_ptr<INetwork> net::create()
	{
		return std::unique_ptr<net>();
	}

	std::unique_ptr<INetwork> net::copy()
	{
		return std::unique_ptr<net>(new net{ *this });
	}
}