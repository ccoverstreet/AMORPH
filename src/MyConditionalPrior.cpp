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
    max_width = exp(log(0.001*x_range) + log(100.0)*rng.rand());
}

double MyConditionalPrior::perturb_hyperparameters(DNest4::RNG& rng)
{
    double logH = 0.0;

    int which = rng.rand_int(4);
    if(which == 0)
    {
        logH += laplace.perturb(location_log_amplitude, rng);
    }
    else if(which == 1)
    {
        scale_log_amplitude += 5*rng.randh();
        DNest4::wrap(scale_log_amplitude, 0.0, 5.0);
    }
    else if(which == 2)
    {
        K += rng.randh();
        DNest4::wrap(K, 0.0, 1.0);
    }
    else
    {
        max_width = log(max_width);
        max_width += log(100.0)*rng.randh();
        DNest4::wrap(max_width, log(0.001*x_range), log(0.1*x_range));
        max_width = exp(max_width);
    }

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

    double logp = 0.0;

    logp += l.log_pdf(vec[1]);
    logp += -log(max_width - min_width);

    return logp;
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

