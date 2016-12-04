#include "ConditionalPriorSinewaves.h"
#include "DNest4/code/DNest4.h"
#include <cmath>

using namespace DNest4;

namespace Crystals
{

const DNest4::Cauchy ConditionalPriorSinewaves::cauchy(0.0, 5.0);

ConditionalPriorSinewaves::ConditionalPriorSinewaves()
{

}

void ConditionalPriorSinewaves::from_prior(RNG& rng)
{
    location_log_period = cauchy.generate(rng);
    scale_log_period = 5*rng.rand();

    location_log_amplitude = cauchy.generate(rng);
    scale_log_amplitude = 5*rng.rand();
}

double ConditionalPriorSinewaves::perturb_hyperparameters(RNG& rng)
{
	double logH = 0.0;

    int which = rng.rand_int(4);

    if(which == 0)
    {
        logH += cauchy.perturb(location_log_period, rng);
    }
    else if(which == 1)
    {
        scale_log_period += 5.0*rng.randh();
        wrap(scale_log_period, 0.0, 5.0);
    }
    else if(which == 2)
    {
        logH += cauchy.perturb(location_log_amplitude, rng);
    }
    else
    {
        scale_log_amplitude += 5.0*rng.randh();
        wrap(scale_log_amplitude, 0.0, 5.0);
    }

	return logH;
}

// vec[0] = log period
// vec[1] = log amplitude
// vec[2] = phase

double ConditionalPriorSinewaves::log_pdf(const std::vector<double>& vec) const
{
    double logp = 0.0;

    Laplace l1(location_log_period, scale_log_period);
    Laplace l2(location_log_amplitude, scale_log_amplitude);

    logp += l1.log_pdf(vec[0]);
    logp += l2.log_pdf(vec[1]);
    if(vec[2] < 0.0 || vec[2] > 2*M_PI)
        logp = -1E300;
    else
        logp += -log(2*M_PI);

    return logp;
}

void ConditionalPriorSinewaves::from_uniform(std::vector<double>& vec) const
{
    Laplace l1(location_log_period, scale_log_period);
    Laplace l2(location_log_amplitude, scale_log_amplitude);

    vec[0] = l1.cdf_inverse(vec[0]);
    vec[1] = l2.cdf_inverse(vec[1]);
    vec[2] = 2*M_PI*vec[2];
}

void ConditionalPriorSinewaves::to_uniform(std::vector<double>& vec) const
{
    Laplace l1(location_log_period, scale_log_period);
    Laplace l2(location_log_amplitude, scale_log_amplitude);

    vec[0] = l1.cdf(vec[0]);
    vec[1] = l2.cdf(vec[1]);
    vec[2] = vec[2]/(2*M_PI);
}

void ConditionalPriorSinewaves::print(std::ostream& out) const
{
    out<<location_log_period<<' '<<scale_log_period<<' ';
    out<<location_log_amplitude<<' '<<scale_log_amplitude<<' ';
}

void ConditionalPriorSinewaves::read(std::istream& in)
{
    in>>location_log_period>>scale_log_period;
    in>>location_log_amplitude>>scale_log_amplitude;
}

} // namespace Crystals

