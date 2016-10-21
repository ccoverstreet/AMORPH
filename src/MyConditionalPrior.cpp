#include "MyConditionalPrior.h"
#include "DNest4/code/DNest4.h"
#include <cmath>
#include <stdexcept>

namespace Crystals
{

const DNest4::Laplace MyConditionalPrior::laplace(0.0, 5.0);

MyConditionalPrior::MyConditionalPrior(double x_min, double x_max)
:x_min(x_min)
,x_max(x_max)
,x_range(x_max - x_min)
{
    if(x_min >= x_max)
        throw std::domain_error("Error in MyConditionalPrior constructor.");
}

void MyConditionalPrior::from_prior(DNest4::RNG& rng)
{

}

double MyConditionalPrior::perturb_hyperparameters(DNest4::RNG& rng)
{
    double logH = 0.0;

    return logH;
}

// {center, log_amplitude, width}

double MyConditionalPrior::log_pdf(const std::vector<double>& vec) const
{
    if(vec[0] < x_min || vec[0] > x_max)
        return -1E300;

    if(vec[2] < min_width || vec[2] > max_width)
        return -1E300;

    DNest4::Laplace l(location_log_amplitude, scale_log_amplitude);
    return l.log_pdf(vec[1]);
}

void MyConditionalPrior::from_uniform(std::vector<double>& vec) const
{

}

void MyConditionalPrior::to_uniform(std::vector<double>& vec) const
{

}

void MyConditionalPrior::print(std::ostream& out) const
{
    out<<' ';
}

} // namespace Crystals

