#include "pch.h"
#include "Mash2D.h"

namespace mash2d
{
	double acson::think()const
	{
		return m_pInput->get_out() * m_W;
	}

	void acson::back_propagate(double err) const { m_pInput->accum_error(m_W * err); }

	void acson::change(double db) { m_W += db * m_pInput->get_out(1); }

	neuron::neuron()
		: fired{ false }
	{
	}

	void neuron::connect(neuron& inp)
	{

#ifdef _DEBUG
		for (auto& a : acsons)
			assert(!a.same(&inp));
#endif // _DEBUG

		acsons.emplace_back(inp);
	}

	void neuron::set_out(double val)
	{
		out = val;
		fired = true;
	}

	void neuron::think()
	{
		auto val{ bias };

		for (auto& a : acsons)
			val += a.think();

		set_out(tanh(val));
	}

	void neuron::ibaw()
	{
		if (auto const ac{ acsons.size() })
		{
			auto stddev{ std::sqrt(2. / ac) };
			bias = rnd() * stddev;

			for (auto& a : acsons)
				a.set_weight(rnd() * stddev);
		}
	}

	void neuron::set_real_error(double db)
	{
		err = db;

		for (auto& a : acsons)
			a.back_propagate(err);

		fired = false;
	}

	void neuron::propagate_error()
	{
		err = err * der(out);

		for (auto& a : acsons)
			a.back_propagate(err);

		fired = false;
	}

	void neuron::adjust()
	{
		const auto bit{ learning_coo * err };
		bias += bit;

		for (auto& a : acsons)
			a.change(bit);
	}

	layer::layer(int neurons)
		:cells(neurons)
	{
	}

	size_t layer::side() const
	{
		return static_cast<size_t>(std::sqrt(cells.size()));
	}

	neuron& layer::get_cell(size_t ind)
	{
		return cells.at(ind);
	}

	void layer::set_input(cvd& inp)
	{
		auto iter{ cells.begin() };

		for (auto val : inp)
			(*iter++).set_out(val);
	}

	void layer::init_biases_and_weights()
	{
		for (auto& n : cells)
			n.ibaw();
	}

	void net::init(std::vector<int> const& topology)
	{
		for (size_t i = 0; i < topology.size() - 1; i++)
		{
			auto root{ static_cast<int>(std::sqrt(topology[i])) };
			assert(sq(root) == topology[i]);
		}

		//m_Input.resize(topology.front());
		m_Output.resize(topology.back());

		for (auto iter{ topology.begin() }; iter != topology.end(); ++iter)
			m_Layers.emplace_back(*iter);

		//	connect neurons
		for (auto itLay{ std::next(m_Layers.begin()) };
			itLay != std::prev(m_Layers.end());
			itLay++)
		{
			auto laySide{ static_cast<int>(itLay->side()) };
			auto itSrc{ itLay };
			auto cur_acson_length{ acson_length };

			do
			{
				itSrc = std::prev(itSrc);
				auto inpSide{ static_cast<int>(itSrc->side()) };

				for (int y = 0; y < laySide; ++y)
					for (int x = 0; x < laySide; ++x)
					{
						auto& n = itLay->get_cell(y * laySide + x);
						auto _y{ y * inpSide / laySide };
						auto _x{ x * inpSide / laySide };

						auto const y_from{ std::max(0, _y - acson_length) },
							y_to{ std::min(inpSide - 1, _y + acson_length) },
							x_from{ std::max(0, _x - acson_length) },
							x_to{ std::min(inpSide - 1, _x + acson_length) };

						for (int iY = y_from; iY <= y_to; ++iY)
							for (int iX = x_from; iX <= x_to; ++iX)
								n.connect(itSrc->get_cell(iY * inpSide + iX));
					}

			} while (--cur_acson_length > 0
				&& itSrc != m_Layers.begin());
		}

		auto itb4Last{ std::next(m_Layers.rbegin()) };

		for (auto& dst : m_Layers.back().get_cells())
			for (auto& scr : itb4Last->get_cells())
				dst.connect(scr);

		for (auto& l : m_Layers)
			l.init_biases_and_weights();
	}

	cvd& net::think(cvd& inp)
	{
		auto itLay{ m_Layers.begin() };
		itLay->set_input(inp);

		for (itLay++; itLay != m_Layers.end(); itLay++)
			for (auto& n : itLay->get_cells())
				n.think();

		/*	if (m_exeLine.empty())
		{
		}
		else
		{
			for (auto p : m_exeLine)
				p->think();
		}*/

		auto itOut{ m_Output.begin() };

		for (auto& n : m_Layers.back().get_cells())
			*itOut++ = n.get_out();

		assert_all_neurons_fired(true);

		return m_Output;
	}

	double net::learn(cvd& real)
	{
		for (auto& l : m_Layers)
			for (auto& n : l.get_cells())
				n.reset_error();

		auto err{ real };

		auto itOut{ m_Output.begin() };

		for (auto& val : err)
		{
			val -= *itOut++;
			val /= 2.;
		}

		auto itErr{ err.begin() };

		for (auto& n : m_Layers.back().get_cells())
			n.set_real_error(*itErr++);

		for (auto iter{ std::next(m_Layers.rbegin()) }; iter != m_Layers.rend(); ++iter)
			for (auto& n : iter->get_cells())
				n.propagate_error();

		assert_all_neurons_fired(false);

		for (auto& l : m_Layers)
			for (auto& n : l.get_cells())
				n.adjust();

		m_Error = .0;

		for (auto val : err)
			m_Error += val * val;

		return m_Error;
	}

	double net::error() const
	{
		return m_Error;
	}

	std::vector<size_t> net::topology() const
	{
		return std::vector<size_t>();
	}

	void net::Serialize(std::ostream&)
	{
	}

	void net::Serialize(std::istream&)
	{
	}

	void net::assert_all_neurons_fired(bool bVal) const
	{
#ifdef _DEBUG
		for (auto& l : m_Layers)
			for (auto& n : l.get_cells())
				assert(n.is_fired() == bVal);
#endif // _DEBUG
	}

	std::unique_ptr<INetwork> net::create()
	{
		return {};
	}

	std::unique_ptr<INetwork> net::copy()
	{
		return {};
	}
}