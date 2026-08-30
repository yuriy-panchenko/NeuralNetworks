#include "pch.h"
#include "Mash.h"
#include <cmath>
#include <algorithm>

namespace mash
{
	void net::init(std::vector<int> const& topology)
	{
		assert(topology.size() > 2);
		err = .0;
		input.init(topology.front());
		output.init(topology.back());

		auto total{ 0 };

		for (size_t i = 1; i < topology.size() - 1; i++)
			total += topology[i];

		auto side{ pow(total, 1. / 3.) };
		iSide = static_cast<int>(side) + 1;
		total = static_cast<int>(iSide * iSide * iSide);

		cube.resize(iSide, Matrix(iSide, Stride(iSide)));

		//	connect input sides
		/*for (size_t y = 0; y < iSide; ++y)
			for (size_t x = 0; x < iSide; ++x)
				cube[0][y][x].connect(input.get_skin());

		for (size_t z = 0; z < iSide; ++z)
			for (size_t x = 0; x < iSide; ++x)
				cube[z][0][x].connect(input.get_skin());

		for (size_t z = 0; z < iSide; ++z)
			for (size_t y = 0; y < iSide; ++y)
				cube[z][y][0].connect(input.get_skin());*/


				//	connect output sides
		for (size_t y = 0; y < iSide; ++y)
			for (size_t x = 0; x < iSide; ++x)
				output.connect(&cube[iSide - 1][y][x]);

		for (size_t z = 0; z < iSide; ++z)
			for (size_t x = 0; x < iSide; ++x)
				output.connect(&cube[z][iSide - 1][x]);

		for (size_t z = 0; z < iSide; ++z)
			for (size_t y = 0; y < iSide; ++y)
				output.connect(&cube[z][y][iSide - 1]);

		assert(iSide);
		//	wire the cube
		for (size_t z = 0; z < iSide; ++z)
			for (size_t y = 0; y < iSide; ++y)
				for (size_t x = 0; x < iSide; ++x)
					wire_cube((int)x, (int)y, (int)z);

		for (size_t z = 0; z < iSide; ++z)
			for (size_t y = 0; y < iSide; ++y)
				for (size_t x = 0; x < iSide; ++x)
					if (z && y && x)
						cube[z][y][x].init();
					else cube[z][y][x].init((int)input.size());

		output.correct_initial();
	}

	cvd& net::think(cvd& inp)
	{
		assert(inp.size() == input.size());
		input.forward(inp);

		for (size_t y = 0; y < iSide; ++y)
			for (size_t x = 0; x < iSide; ++x)
				cube[0][y][x].think(input.get_plane(0));

		for (size_t z = 0; z < iSide; ++z)
			for (size_t x = 0; x < iSide; ++x)
				cube[z][0][x].think(input.get_plane(1));

		for (size_t z = 0; z < iSide; ++z)
			for (size_t y = 0; y < iSide; ++y)
				cube[z][y][0].think(input.get_plane(2));

		auto get_waiting = [this](auto& ret)
			{
				ret.clear();
				for (auto& m : cube)
					for (auto& l : m)
						for (auto& n : l)
							if (n.is_waiting())
								ret.push_back(&n);
				return !ret.empty();
			};

		auto filter_ready = [](std::vector<neuron*>& ret)
			{
				for (int i = 0; i < (int)ret.size(); i++)
					if (!ret[i]->ready())
					{
						if (i < ret.size() - 1)
						{
							std::swap(ret[i], ret[ret.size() - 1]);
							--i;
						}
						ret.erase(std::prev(ret.end()));
					}
			};

		//	process cube
		std::vector<neuron*> ns;
		while (get_waiting(ns))
		{
			filter_ready(ns);
			assert(!ns.empty());
			for (auto p : ns)
				p->think();
		}

		output.think();
		return output.out();
	}

	double net::learn(vd const& real)
	{
		err = output.learn(real);

		//	learn cube
		std::vector<neuron*> ns;
		do
		{
			ns = get_eager();
			for (auto p : ns)
				p->learn();
		} while (!ns.empty());

		for (size_t z = 0; z < iSide; ++z)
			for (size_t y = 0; y < iSide; ++y)
				for (size_t x = 0; x < iSide; ++x)
					if (z && y && x)
						cube[z][y][x].change_weights();
					else cube[z][y][x].change_weights(z ? (y ? input.get_plane(2) : input.get_plane(1)) : input.get_plane(0));

		output.change_weights();

		return error();
	}

	double net::error() const
	{
		return err;
	}

	std::vector<size_t> net::topology() const
	{
		return { input.size(), iSide * iSide * iSide, output.size() };
	}

	void net::Serialize(std::ostream&)
	{
	}

	void net::Serialize(std::istream&)
	{
	}

	std::unique_ptr<INetwork> net::create() { return std::make_unique<net>(); }

	std::unique_ptr<INetwork> net::copy() { return std::make_unique<net>(*this); }

	//////////////////////////////////////////////////////////////////////////////////////////

	void neuron::init()
	{
		auto const stddev{ dev(ins.size()) };
		bias = rnd() * stddev;

		for (auto& w : ws)
			w *= stddev;
	}

	void neuron::init(int cells)
	{
		auto const stddev{ dev(cells) };
		bias = rnd() * stddev;
		ws.resize(cells);

		for (auto& val : ws)
			val = rnd() * stddev;
	}

	void neuron::connect(std::vector<neuron>& ns)
	{
		for (auto& n : ns)
			connect(&n);
	}

	void neuron::connect(neuron* pCell)
	{
		ins.push_back(pCell);
		ws.push_back(rnd());
		pCell->outs.push_back(this);
	}

	double neuron::think()
	{
		vd inp(ins.size());
		auto itInp{ inp.begin() };
		for (auto pN : ins)
			*itInp++ = pN->dOut;
		think(inp);
		return dOut;
	}

	void neuron::think(cvd& data)
	{
		assert(data.size() == ws.size());

		auto res{ bias };
		auto itInp{ data.begin() };

		for (auto w : ws)
			res += w * *itInp++;

		dOut = tanh(res);
		fired = true;
	}

	bool neuron::is_waiting() const
	{
		if (fired)
			return false;
		return !ins.empty();
	}

	bool neuron::ready() const
	{
		for (auto p : ins)
			if (!p->fired)
				return false;
		return !ins.empty();
	}

	void neuron::set_error(double x)
	{
		err = x;
		fired = false;
	}

	bool neuron::is_eager() const
	{
		if (!fired)
			return false;

		for (auto p : outs)
			if (p->fired)
				return false;

		return true;
	}

	void neuron::learn()
	{
		assert(fired);

		err = .0;

		for (auto prev : outs)
			err += prev->weighted_error(this);

		set_error(err * der(dOut));
	}

	double neuron::weighted_error(neuron* p) const
	{
		auto it{ std::find(ins.begin(),ins.end(),p) };
		assert(it != ins.end());
		return ws[std::distance(ins.begin(), it)] * err;
	}

	void neuron::change_weights()
	{
		auto itIns{ ins.begin() };
		auto const dErr{ learning_rate * err };
		bias -= dErr;

		for (auto& w : ws)
			w -= dErr * (*itIns++)->out();
	}

	void neuron::change_weights(cvd& inp)
	{
		auto itIns{ inp.begin() };
		auto const dErr{ learning_rate * err };
		bias -= dErr;

		for (auto& w : ws)
			w -= dErr * *itIns++;
	}

	void encoder::init(int i)
	{
		for (auto& v : plane)
			v.resize(i);
	}

	void decoder::connect(neuron* p)
	{
		for (auto& n : ins)
			n.connect(p);
	}

	void decoder::think()
	{
		for (size_t iOut = 0; iOut < result.size(); iOut++)
			result[iOut] = ins[iOut].think();
	}

	void net::wire_cube(int x, int y, int z)
	{
		auto const
			x_from{ std::max(0, x - max_acson_length) },
			y_from{ std::max(0, y - max_acson_length) },
			z_from{ std::max(0, z - max_acson_length) };

		for (int _z = z_from; _z < z; ++_z)
			for (int _y = y_from; _y < y; ++_y)
				for (int _x = x_from; _x < x; ++_x)
					cube[z][y][x].connect(&cube[_z][_y][_x]);
	}

	std::vector<neuron*> net::get_eager()
	{
		std::vector<neuron*> ret;;

		for (auto& s : cube)
			for (auto& l : s)
				for (auto& n : l)
					if (n.is_eager())
						ret.push_back(&n);
		return ret;
	}

	double decoder::learn(cvd& real)
	{
		double ret{ .0 }, err;

		for (size_t i = 0; i < ins.size(); ++i)
		{
			err = (result[i] - real[i]) / 2.;
			ins[i].set_error(err * der(result[i]));
			ret += sq(err);
		}
		return ret;// / ins.size();
	}

	void decoder::correct_initial()
	{
		for (auto& n : ins)
			n.init();
	}

	void decoder::change_weights()
	{
		for (auto& n : ins)
			n.change_weights();
	}

	void encoder::forward(cvd& v)
	{
		//	0
		std::copy(v.begin(), v.end(), plane[0].begin());


		auto img = [&v](int x, int y)->double
			{
				return v[std::clamp(y, 0, 27) * 28 + std::clamp(x, 0, 27)];
			};

		auto aver = [img](int const x, int const y)->double
			{
				double ret{};

				for (int _y{ y - 1 }; _y <= y + 1; ++_y)
					for (int _x{ x - 1 }; _x <= x + 1; ++_x)
						ret += img(_x, _y);

				return ret / 9.;
			};

		//	1
		//	2
		for (int y = 0; y < 28; ++y)
			for (int x = 0; x < 28; ++x)
			{
				plane[1][y * 28 + x] = std::sqrt(sq(img(x + 1, y) - img(x - 1, y)) + sq(img(x, y + 1) - img(x, y - 1)));
				plane[2][y * 28 + x] = aver(x, y);
			}

	}
}