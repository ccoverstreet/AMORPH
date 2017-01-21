#include "MyConditionalPrior.h"
#include "DNest4/code/DNest4.h"
#include <cmath>
#include <stdexcept>

namespace Crystals
{

const DNest4::Laplace MyConditionalPrior::laplace(0.0, 5.0);

MyConditionalPrior::MyConditionalPrior(double x_min, double x_max,
    bool narrow)
:x_min(x_min)
,x_max(x_max)
,x_range(x_max - x_min)
,narrow(narrow)
{
    if(x_min >= x_max)
        throw std::domain_error("Error in MyConditionalPrior constructor.");
}

void MyConditionalPrior::from_prior(DNest4::RNG& rng)
{
    location_log_amplitude = laplace.generate(rng);
    scale_log_amplitude = 5*rng.rand();

    if(narrow)
    {
        location_log_width = log(0.001*x_range) + log(50.0)*rng.rand();
        scale_log_width = 0.1*rng.rand();
    }
    else
    {
        location_log_width = log(0.05*x_range) + log(20.0)*rng.rand();
        scale_log_width = 0.2*rng.rand();
    }
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
        if(narrow)
        {
            location_log_width += log(50.0)*rng.randh();
            DNest4::wrap(location_log_width, log(0.001*x_range), log(0.05*x_range));
        }
        else
        {
            location_log_width += log(20.0)*rng.randh();
            DNest4::wrap(location_log_width, log(0.05*x_range), log(x_range));
        }

    }
    else
    {
        if(narrow)
        {
            scale_log_width += 0.1*rng.randh();
            DNest4::wrap(scale_log_width, 0.0, 0.1);
        }
        else
        {
            scale_log_width += 0.2*rng.randh();
            DNest4::wrap(scale_log_width, 0.0, 0.2);
        }
    }

    return logH;
}

// {center, log_amplitude, width}

double MyConditionalPrior::log_pdf(const std::vector<double>& vec) const
{
    if(narrow)
    {
        if(vec[0] < x_min || vec[0] > x_max)
            return -1E300;
    }
    else
    {
        double xc = 0.5*(x_min + x_max);
        if(std::abs(vec[0] - xc) > 0.2*x_range)
            return -1E300;
    }

    if(vec[2] <= 0.0)
        return -1E300;

    DNest4::Laplace l(location_log_amplitude, scale_log_amplitude);

    double logp = 0.0;

    logp += l.log_pdf(vec[1]);

    l = DNest4::Laplace(location_log_width, scale_log_width);
    logp += -log(vec[2]) + l.log_pdf(log(vec[2]));

    return logp;
}

void MyConditionalPrior::from_uniform(std::vector<double>& vec) const
{
    DNest4::Laplace l(location_log_amplitude, scale_log_amplitude);

    if(narrow)
    {
        vec[0] = x_min + x_range*vec[0];
    }
    else
    {
        double xc = 0.5*(x_min + x_max);
        vec[0] = xc - 0.2*x_range + 0.4*x_range*vec[0];
    }

    vec[1] = l.cdf_inverse(vec[1]);

    l = DNest4::Laplace(location_log_width, scale_log_width);
    vec[2] = exp(l.cdf_inverse(vec[2]));
}

void MyConditionalPrior::to_uniform(std::vector<double>& vec) const
{
    DNest4::Laplace l(location_log_amplitude, scale_log_amplitude);

    if(narrow)
    {
        vec[0] = (vec[0] - x_min)/x_range;
    }
    else
    {
        double xc = 0.5*(x_min + x_max);
        vec[0] = (vec[0] - (xc - 0.2*x_range)) / (0.4 * x_range);
    }
    vec[1] = l.cdf(vec[1]);

    l = DNest4::Laplace(location_log_width, scale_log_width);
    vec[2] = l.cdf(log(vec[2]));
}

void MyConditionalPrior::print(std::ostream& out) const
{
    out<<location_log_amplitude<<' '<<scale_log_amplitude<<' ';
    out<<location_log_width<<' '<<scale_log_width<<' ';
}

} // namespace Crystals

