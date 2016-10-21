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
    location_log_amplitude = laplace.generate(rng);
    scale_log_amplitude = 5*rng.rand();

    K = rng.rand();
    max_width = exp(log(1E-3*x_range) + log(1E3)*rng.rand());
}

double MyConditionalPrior::perturb_hyperparameters(DNest4::RNG& rng)
{
    double logH = 0.0;

    return logH;
}

// {center, log_amplitude, width}

double MyConditionalPrior::log_pdf(const std::vector<double>& vec) const
{
    double min_width = K*max_width;

    if(vec[0] < x_min || vec[0] > x_max)
        return -1E300;

    if(vec[2] < min_width || vec[2] > max_width)
        return -1E300;

    DNest4::Laplace l(location_log_amplitude, scale_log_amplitude);
    return l.log_pdf(vec[1]);
}

void MyConditionalPrior::from_uniform(std::vector<double>& vec) const
{
    double min_width = K*max_width;

    DNest4::Laplace l(location_log_amplitude, scale_log_amplitude);
    vec[0] = x_min + x_range*vec[0];
    vec[1] = l.cdf_inverse(vec[1]);
    vec[2] = min_width + (max_width - min_width)*vec[2];
}

void MyConditionalPrior::to_uniform(std::vector<double>& vec) const
{
    double min_width = K*max_width;

    DNest4::Laplace l(location_log_amplitude, scale_log_amplitude);
    vec[0] = (vec[0] - x_min)/x_range;
    vec[1] = l.cdf(vec[1]);
    vec[2] = (vec[2] - min_width)/(max_width - min_width);
}

void MyConditionalPrior::print(std::ostream& out) const
{
    out<<location_log_amplitude<<' '<<scale_log_amplitude<<' ';
    out<<K<<' '<<max_width<<' ';
}

} // namespace Crystals

