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

    background = exp(cauchy.generate(rng));

    amplitude = exp(cauchy.generate(rng));
    center = data.get_x_min() + data.get_x_range()*rng.rand();
    width = data.get_x_range()*rng.rand();
}

double MyModel::perturb(DNest4::RNG& rng)
{
    double logH = 0.0;

    int which = rng.rand_int(4);

    if(which == 0)
    {
        background = log(background);
        logH += cauchy.perturb(background, rng);
        background = exp(background);
    }
    else if(which == 1)
    {
        amplitude = log(amplitude);
        logH += cauchy.perturb(amplitude, rng);
        amplitude = exp(amplitude);
    }
    else if(which == 2)
    {
        center += data.get_x_range()*rng.rand();
        DNest4::wrap(center, data.get_x_min(), data.get_x_max());
    }
    else
    {
        width += data.get_x_range()*rng.rand();
        DNest4::wrap(width, 0.0, data.get_x_range());
    }

    return logH;
}

double MyModel::log_likelihood() const
{
    double logL = 0.0;

    // References to the data vectors
    const auto& data_x = data.get_x();
    const auto& data_y = data.get_y();

    if(std::isnan(logL) || std::isinf(logL))
        logL = -1E300;

    return logL;
}

void MyModel::print(std::ostream& out) const
{
    out<<background<<' '<<amplitude<<' '<<center<<' '<<width<<' ';
}

std::string MyModel::description() const
{
    return std::string("background, amplitude, center, width, ");
}

void MyModel::load_data(const char* filename)
{
    data.load(filename);
}

} // namespace Crystals

