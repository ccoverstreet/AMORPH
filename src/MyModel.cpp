#include "MyModel.h"
#include "DNest4/code/DNest4.h"
#include <sstream>

namespace Crystals
{

Data MyModel::data;
const DNest4::Cauchy MyModel::cauchy(0.0, 5.0);

MyModel::MyModel()
:model_curve(data.get_y().size())
{
    if(!data.get_loaded())
        std::cerr<<"# WARNING: it appears no data has been loaded."<<std::endl;
}

void MyModel::from_prior(DNest4::RNG& rng)
{
    background = exp(cauchy.generate(rng));

    amplitude = exp(cauchy.generate(rng));
    center = data.get_x_min() + data.get_x_range()*rng.rand();
    width = data.get_x_range()*rng.rand();

    sigma0 = exp(cauchy.generate(rng));
    sigma1 = exp(cauchy.generate(rng));
    nu = exp(log(1.0) + log(1E3)*rng.rand());

    compute_model_curve();
}

double MyModel::perturb(DNest4::RNG& rng)
{
    double logH = 0.0;

    int which = rng.rand_int(7);

    if(which == 0)
    {
        background = log(background);
        logH += cauchy.perturb(background, rng);
        background = exp(background);

        compute_model_curve();
    }
    else if(which == 1)
    {
        amplitude = log(amplitude);
        logH += cauchy.perturb(amplitude, rng);
        amplitude = exp(amplitude);

        compute_model_curve();
    }
    else if(which == 2)
    {
        center += data.get_x_range()*rng.rand();
        DNest4::wrap(center, data.get_x_min(), data.get_x_max());

        compute_model_curve();
    }
    else if(which == 3)
    {
        width += data.get_x_range()*rng.rand();
        DNest4::wrap(width, 0.0, data.get_x_range());

        compute_model_curve();
    }
    else if(which == 4)
    {
        sigma0 = log(sigma0);
        logH += cauchy.perturb(sigma0, rng);
        sigma0 = exp(sigma0);
    }
    else if(which == 5)
    {
        sigma1 = log(sigma1);
        logH += cauchy.perturb(sigma1, rng);
        sigma1 = exp(sigma1);
    }
    else
    {
        nu = log(nu);
        nu += log(1E3)*rng.randh();
        DNest4::wrap(nu, log(1.0), log(1E3));
        nu = exp(nu);
    }

    return logH;
}

void MyModel::compute_model_curve()
{
    const auto& data_x = data.get_x();
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
    out<<sigma0<<' '<<sigma1<<' '<<nu<<' ';
}

std::string MyModel::description() const
{
    std::stringstream s;
    s<<"background, amplitude, center, width, ";
    s<<"sigma0, sigma1, nu, ";
    return s.str();
}

void MyModel::load_data(const char* filename)
{
    data.load(filename);
}

} // namespace Crystals

