#include "MyModel.h"
#include "DNest4/code/DNest4.h"

namespace Crystals
{

MyModel::MyModel()
{

}

void MyModel::from_prior(DNest4::RNG& rng)
{
    const DNest4::Cauchy cauchy(0.0, 5.0);

    background = cauchy.generate(rng);
}

double MyModel::perturb(DNest4::RNG& rng)
{
    double logH = 0.0;

    const DNest4::Cauchy cauchy(0.0, 5.0);

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

