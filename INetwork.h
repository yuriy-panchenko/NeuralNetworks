#pragma once
#include <vector>

class INetwork abstract
{
public:
	using vd = std::vector<double>;
	using cvd = std::vector<double> const;

public:
	virtual ~INetwork() = default;

	virtual void init(std::vector<int> const& topology)abstract;
	virtual cvd & think(cvd&)abstract;
	virtual double learn(vd const&)abstract;
	virtual double error()const abstract;
	virtual std::vector<size_t> topology()const abstract;

	virtual std::unique_ptr<INetwork> create()abstract;
	virtual std::unique_ptr<INetwork> copy()abstract;

protected:
	virtual void Serialize(std::ostream&)abstract;
	virtual void Serialize(std::istream&)abstract;

public:
	friend std::ostream& operator<<(std::ostream& s, INetwork& n) { n.Serialize(s); return s; }
	friend std::istream& operator>>(std::istream& s, INetwork& n) { n.Serialize(s); return s; }
};