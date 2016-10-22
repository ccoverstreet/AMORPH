#include "MyModel.h"
#include "DNest4/code/DNest4.h"
#include <sstream>
#include <cmath>

namespace Crystals
{

Data MyModel::data;
const DNest4::Laplace MyModel::laplace(0.0, 5.0);

MyModel::MyModel()
:spikes(3, max_num_spikes, false,
            MyConditionalPrior(data.get_x_min(), data.get_x_max()),
                                DNest4::PriorType::log_uniform)
,model_curve(data.get_y().size())
{
    if(!data.get_loaded())
        std::cerr<<"# WARNING: it appears no data has been loaded."<<std::endl;
}

void MyModel::from_prior(DNest4::RNG& rng)
{
    background = exp(laplace.generate(rng));

    amplitude = exp(laplace.generate(rng));
    center = data.get_x_min() + data.get_x_range()*rng.rand();
    width = data.get_x_range()*rng.rand();

    spikes.from_prior(rng);

    sigma0 = exp(laplace.generate(rng));
    sigma1 = exp(laplace.generate(rng));
    nu = exp(log(1.0) + log(1E3)*rng.rand());

    compute_model_curve();
}

double MyModel::perturb(DNest4::RNG& rng)
{
    double logH = 0.0;

    // Select blocks of parameters
    int choice = rng.rand_int(3);

    if(choice == 0)
    {
        // Perturb spikes
        logH += spikes.perturb(rng);

        compute_model_curve();
    }
    else if(choice == 1)
    {
        // Perturb a parameter related to the background or wide component
        int which = rng.rand_int(4);

        if(which == 0)
        {
            background = log(background);
            logH += laplace.perturb(background, rng);
            background = exp(background);

            compute_model_curve();
        }
        else if(which == 1)
        {
            amplitude = log(amplitude);
            logH += laplace.perturb(amplitude, rng);
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
    }
    else
    {
        // Perturb a noise-related parameter
        int which = rng.rand_int(3);

        if(which == 0)
        {
            sigma0 = log(sigma0);
            logH += laplace.perturb(sigma0, rng);
            sigma0 = exp(sigma0);
        }
        else if(which == 1)
        {
            sigma1 = log(sigma1);
            logH += laplace.perturb(sigma1, rng);
            sigma1 = exp(sigma1);
        }
        else
        {
            nu = log(nu);
            nu += log(1E3)*rng.randh();
            DNest4::wrap(nu, log(1.0), log(1E3));
            nu = exp(nu);
        }
    }

    return logH;
}

void MyModel::compute_model_curve()
{
    const auto& data_x = data.get_x();

    double tau = 1.0/(width*width);
    double rsq;

    for(size_t i=0; i<model_curve.size(); ++i)
    {
        rsq = pow(data_x[i] - center, 2);
        model_curve[i] = amplitude*exp(-0.5*rsq*tau);
    }

    double c, a, w;
    const auto& components = spikes.get_components();
    for(size_t i=0; i<components.size(); ++i)
    {
        // {center, log_amplitude, width}
        c = components[i][0];
        a = exp(components[i][1]);
        w = components[i][2];
        tau = 1.0/(w*w);

        for(size_t j=0; j<model_curve.size(); ++j)
        {
            rsq = pow(data_x[j] - c, 2);
            if(rsq*tau < 100.0)
                model_curve[j] += a*exp(-0.5*rsq*tau);
        }
    }
}

double MyModel::log_likelihood() const
{
    double logL = 0.0;

    // References to the data vector
    const auto& data_y = data.get_y();

    // Normalising constant of the t distribution
    double C = lgamma(0.5*(nu + 1.0)) - log(0.5*nu) - 0.5*log(M_PI*nu);

    // T likelihood
    double resid, var;
    for(size_t i=0; i<data_y.size(); ++i)
    {
        resid = data_y[i] - model_curve[i];
        var = sigma0*sigma0 + sigma1*model_curve[i];

        logL += C - 0.5*log(var)
                  - 0.5*(nu + 1.0)*log(1.0 + resid*resid/var);
    }

    if(std::isnan(logL) || std::isinf(logL))
        logL = -1E300;

    return logL;
}

void MyModel::print(std::ostream& out) const
{
    out<<background<<' '<<amplitude<<' '<<center<<' '<<width<<' ';
    spikes.print(out);
    out<<sigma0<<' '<<sigma1<<' '<<nu<<' ';

    for(auto m: model_curve)
        out<<m<<' ';
}

std::string MyModel::description() const
{
    std::stringstream s;
    s<<"background, amplitude, center, width, ";
    s<<"dim_spikes, max_num_spikes, ";
    s<<"location_log_amplitude, scale_log_amplitude, ";
    s<<"K, max_width, num_spikes, ";
    for(size_t i=0; i<max_num_spikes; ++i)
        s<<"center["<<i<<"], ";
    for(size_t i=0; i<max_num_spikes; ++i)
        s<<"log_amplitude["<<i<<"], ";
    for(size_t i=0; i<max_num_spikes; ++i)
        s<<"width["<<i<<"], ";
    s<<"sigma0, sigma1, nu, ";
    for(size_t i=0; i<model_curve.size(); ++i)
        s<<"model_curve["<<i<<"], ";

    return s.str();
}

void MyModel::load_data(const char* filename)
{
    data.load(filename);
}

} // namespace Crystals

