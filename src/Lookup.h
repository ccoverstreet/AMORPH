#ifndef Crystals_Lookup
#define Crystals_Lookup

#include <vector>

/*
* Lookup tables for speeding things up
* Singleton pattern
*/

namespace Crystals
{

class Lookup
{
	private:
		int num;
		double xMin, xMax, dx, one_over_dx;
		std::vector<double> _exp; // exp(-x) for x >= 0

		Lookup();
		Lookup(const Lookup& other);

		static Lookup instance;

	public:
		static double evaluate(double x);

};

} // namespace Crystals

#endif

