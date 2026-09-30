#pragma once

namespace uni
{
	template<typename T = double>
	class random_device
	{
	public:
		static T generate()
		{
			return std::normal_distribution<T>{T(0), T(1)}(rng);
		}
	private:
		inline static std::mt19937 rng{ std::random_device{}() };
	};

	template<typename T>
	class matrix
	{
		size_t m_Cx, m_Cy;
		std::vector<T> m_Data;
	public:
		matrix(size_t acson_count, size_t neuron_count);
		matrix(size_t acsons, size_t neurons, T initial_value);
		matrix(matrix&&) = default;

		T const* row(size_t row_index)const { return ptr() + row_index * m_Cx; }
		T* row(size_t index) { return ptr() + index * m_Cx; }
		T* ptr() { return m_Data.data(); }
		T const* ptr()const { return m_Data.data(); }
		T const* ptr_end()const { return ptr() + m_Cx * m_Cy; }
		T* ptr_end() { return ptr() + m_Cx * m_Cy; }
		void randomize(T scale);
		T product(size_t iRow, std::vector<T> const& inp)const;
		size_t acsons()const { return m_Cx; }
		size_t neurons()const { return m_Cy; }
		void fill(T val) { std::fill(m_Data.begin(), m_Data.end(), val); }
	};

	template<typename activ>
	class layer :
		public std::vector<typename activ::type>,
		public ISerialize
	{
	public:
		using T = typename activ::type;

	public:
		layer() = default;
		layer(layer const&) = default;
		layer(layer&&) = default;
		layer(size_t acsons, size_t neurons);
		// Inherited via ISerialize
		void Serialize(std::ofstream& s) override
		{
			for (size_t i = 0; i < m_Bias.size(); i++)
			{
				s.write(reinterpret_cast<char const*>(&m_Bias[i]), sizeof(T));
				s.write(reinterpret_cast<char const*>(m_Weights.row(i)), m_Weights.acsons() * sizeof(T));
			}
		}
		void Serialize(std::ifstream& s) override
		{
			for (size_t i = 0; i < m_Bias.size(); i++)
			{
				s.read(reinterpret_cast<char*>(&m_Bias[i]), sizeof(T));
				s.read(reinterpret_cast<char*>(m_Weights.row(i)), m_Weights.acsons() * sizeof(T));
			}
		}

		std::vector<T> const& think(std::vector<T> const&);
		std::vector<T> learn(std::vector<T> const& inp, std::vector<T> const&);
		void adjust(T lcoo);
		void shock();

	private:
		std::vector<T> m_Bias, m_dBias;
		matrix<T> m_Weights, m_dWs;
	};

	template<typename T>
	matrix<T>::matrix(size_t acson_count, size_t neuron_count)
		:m_Cx{ acson_count }
		, m_Cy{ neuron_count }
		, m_Data(acson_count* neuron_count)
	{}

	template<typename T>
	matrix<T>::matrix(size_t acsons, size_t neurons, T initial_value)
		:m_Cx{ acsons }
		, m_Cy{ neurons }
		, m_Data(acsons* neurons, initial_value)
	{}


	template<typename T>
	void matrix<T>::randomize(T scale)
	{
		for (auto& val : m_Data)
			val = random_device<T>::generate() * scale;
	}

	template<typename T>
	T matrix<T>::product(size_t iRow, std::vector<T> const& inp) const
	{
		T ret{};
		auto pRow{ row(iRow) };

		for (auto val : inp)
			ret += val * *pRow++;

		return ret;
	}

	template<typename activ>
	layer<activ>::layer(size_t acsons, size_t neurons)
		:std::vector<T>(neurons)
		, m_Weights{ acsons, neurons }
		, m_dWs{ acsons, neurons, T(0) }
		, m_Bias(neurons, T(0))
		, m_dBias(neurons, T(0))
	{
		m_Weights.randomize(std::sqrt(T(2) / acsons));
	}

	template<typename activ>
	std::vector<typename activ::type> const& layer<activ>::think(std::vector<T> const& inp)
	{
		for (size_t i = 0; i < std::vector<T>::size(); ++i)
			(*this)[i] = activ::f(m_Bias[i] + m_Weights.product(i, inp));

		return *this;
	}

	template<typename activ>
	std::vector<typename activ::type> layer<activ>::learn(std::vector<T> const& inp, std::vector<T> const& dL_output)
	{
		std::vector<T> upstream(inp.size(), T(0));   // size = this layer's input dim

		for (size_t iCell = 0; iCell < std::vector<T>::size(); ++iCell)
		{
			auto pWs{ m_Weights.row(iCell) };
			auto pdWs{ m_dWs.row(iCell) };
			auto const delta{ dL_output[iCell] * activ::df((*this)[iCell]) };

			for (size_t iAx = 0; iAx < inp.size(); ++iAx)
			{
				*(pdWs + iAx) += delta * inp[iAx];
				upstream[iAx] = delta * *(pWs + iAx);
			}

			m_dBias[iCell] += delta;
		}

		return upstream;
	}

	template<typename activ>
	void layer<activ>::adjust(T lcoo)
	{
		std::transform(
			m_Weights.ptr(),
			m_Weights.ptr_end(),
			m_dWs.ptr(),
			m_Weights.ptr(),
			[lcoo](T W, T dW) {return W - lcoo * dW; });
		m_dWs.fill(T(0));

		std::transform(
			m_Bias.begin(),
			m_Bias.end(),
			m_dBias.begin(),
			m_Bias.begin(),
			[lcoo](T b, T db) {return b - lcoo * db; });
		std::fill(m_dBias.begin(), m_dBias.end(), T(0));
	}

	template<typename activ_func>
	void layer<activ_func>::shock()
	{
		m_Weights.row(rand() % m_Weights.neurons())[rand() % m_Weights.acsons()] = random_device<T>::generate() * std::sqrt(T(2) / m_Weights.acsons());
	}
}