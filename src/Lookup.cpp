#include "Lookup.h"
#include <cmath>
#include <iostream>

namespace Crystals
{

Lookup Lookup::instance;

Lookup::Lookup()
:num_shape(201)
,shape_min(log(0.1))
,shape_max(log(100.0))
,dshape((shape_max - shape_min)/(num_shape - 1))
,one_over_dshape(1.0/dshape)
,num_x(10001)
,x_min(-20.0)
,x_max(20.0)
,dx((x_max - x_min)/(num_x - 1))
,one_over_dx(1.0/dx)
,f(num_shape, std::vector<double>(num_x))
{
    double nu, x, power;
    for(int i=0; i<num_shape; ++i)
    {
        nu = exp(shape_min + i*dshape);
        power = -0.5*(nu + 1.0);
        for(int j=0; j<num_x; ++j)
        {
            x = x_min + j*dx;
            f[i][j] = pow(1.0 + x*x/nu, power);
        }

        // Normalise numerically
        double tot = 0.0;
        for(int j=0; j<num_x; ++j)
            tot += f[i][j];
        tot *= dx;
        for(int j=0; j<num_x; ++j)
            f[i][j] /= tot;
    }

//    for(int i=0; i<num_shape; ++i)
//    {
//        for(int j=0; j<num_x; ++j)
//        {
//            std::cout << f[i][j] << ' ';
//        }
//        std::cout << std::endl;
//    }
//    exit(0);

}

double Lookup::evaluate(double shape, double x)
{
	int i = (int)((shape - instance.shape_min) * instance.one_over_dshape);
    int j = (int)((x - instance.x_min) * instance.one_over_dx);

    double frac = (x - (instance.x_min + j*instance.dx))*instance.one_over_dx;

	if(i < 0 || i >= instance.num_shape || j < 0 || j >= (instance.num_x - 1))
		return 0.0;

	return frac*instance.f[i][j+1] + (1.0 - frac)*instance.f[i][j];
}

} // namespace Crystals

