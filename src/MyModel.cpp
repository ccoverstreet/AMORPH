#include "MyModel.h"
#include "DNest4/code/DNest4.h"

namespace Crystals
{

Data MyModel::data;
const DNest4::Cauchy MyModel::cauchy(0.0, 5.0);

MyModel::MyModel()
{

}

void MyModel::from_prior(DNest4::RNG& rng)
{
    if(!data.get_loaded())
        std::cerr<<"# WARNING: it appears no data has been loaded."<<std::endl;

    background = cauchy.generate(rng);
}

double MyModel::perturb(DNest4::RNG& rng)
{
    double logH = 0.0;

    int which = rng.rand_int(1);

    if(which == 0)
        logH += cauchy.perturb(background, rng);

    return logH;
}

double MyModel::log_likelihood() const
{
    double logL = 0.0;
    return logL;
}

void MyModel::print(std::ostream& out) const
{
    out<<background<<' ';
}

std::string MyModel::description() const
{
    return std::string("background, ");
}

} // namespace Crystals

