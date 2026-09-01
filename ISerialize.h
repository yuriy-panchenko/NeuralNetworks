#pragma once
#include <fstream>

class ISerialize abstract
{
public:
	friend std::ifstream& operator>>(std::ifstream& s, ISerialize& o)
	{
		o.Serialize(s);
		return s;
	}
	friend std::ofstream& operator<<(std::ofstream& s, ISerialize& o)
	{
		o.Serialize(s);
		return s;
	}
protected:
	virtual void Serialize(std::ofstream&) = 0;
	virtual void Serialize(std::ifstream&) = 0;
};
