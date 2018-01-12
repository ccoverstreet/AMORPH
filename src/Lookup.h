#ifndef Crystals_Lookup
#define Crystals_Lookup

#include <vector>

/*
* Lookup tables for speeding things up
* Singleton pattern
*/

namespace AMORPH
{

class Lookup
{
	private:
		int num_shape;
        double shape_min, shape_max, dshape, one_over_dshape;

        int num_x;
		double x_min, x_max, dx, one_over_dx;
		std::vector<std::vector<double>> f;

		Lookup();
		Lookup(const Lookup& other);

		static Lookup instance;

	public:
		static double evaluate(double shape, double x);

};

} // namespace AMORPH

#endif

