#pragma once
#include <random>
#include "INetwork.h"

namespace proxy
{
	// Simple activation functions
	/*inline double sigmoid(double x)
	{
		return tanh(x);
	}

	inline double sigmoid_derivative(double x)
	{
		return 1. - sq(x);
	}*/
	inline double sigmoid(double x)
	{
		return 1. / (1. + std::exp(-x));
	}

	inline double sigmoid_derivative(double x)
	{
		return x * (1. - x);
	}

	using Vector = INetwork::vd;

	// Neural Network class
	class MultilayerPerceptron
	{
	public:
		using Matrix = std::vector<Vector>;

	private:
		std::vector<int> layers;			// e.g., {2, 4, 1} → input:2, hidden:4, output:1
		std::vector<Matrix> weights;	// weights[i] connects layer i to i+1
		std::vector<Vector> biases,		// biases[i] for layer i+1
			activations;					// activations per layer (including input)

		std::mt19937 rng;
		std::normal_distribution<double> dist;

	public:
		MultilayerPerceptron(const std::vector<int>& layer_sizes);
		MultilayerPerceptron(std::istream&);

		Vector forward(const Vector& input);

		double backprop(const Vector& target, double learning_rate = 0.1);

		void train(const std::vector<Vector>& inputs,
			const std::vector<Vector>& targets,
			int epochs = 10000,
			double lr = 0.1);

		std::vector<size_t> topology()const;
		void save(std::ostream&)const;
	};

	class net
		:public INetwork
	{
		std::unique_ptr<MultilayerPerceptron> pNet;
		double m_Error;
		Vector m_Out;

	public:
		net() = default;
		net(net&&) = default;
		net(net const&);

		net& operator=(net const&);

		void init(std::vector<int> const& topo)override;
		Vector const& think(Vector const&)override;
		double learn(Vector const&)override;
		double error()const override;
		std::vector<size_t> topology()const override;

		// Inherited via INetwork
		void Serialize(std::ostream&) override;
		void Serialize(std::istream&) override;

		// Inherited via INetwork
		std::unique_ptr<INetwork> create() override;
		std::unique_ptr<INetwork> copy() override;
	};
}

