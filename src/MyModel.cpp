#include "MyModel.h"
#include "DNest4/code/DNest4.h"
#include <sstream>
#include <cmath>

namespace Crystals
{

Data MyModel::data;
const DNest4::Laplace MyModel::laplace(0.0, 5.0);

MyModel::MyModel()
:n(data.get_y().size())
,spikes(3, max_num_spikes, false,
            MyConditionalPrior(data.get_x_min(), data.get_x_max()),
                                DNest4::PriorType::log_uniform)
,wide_component(data.get_y().size())
,the_spikes(data.get_y().size())
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

    beta = exp(log(1E-6) + log(1E6)*rng.rand());
    L = exp(log(1E-2)*data.get_x_range() + log(1E2)*rng.rand());
    for(auto& _n: n)
        _n = rng.randn();

    spikes.from_prior(rng);

    sigma0 = exp(laplace.generate(rng));
    sigma1 = exp(laplace.generate(rng));
    nu = exp(log(1.0) + log(1E3)*rng.rand());

    compute_wide_component();
    compute_the_spikes();
}

double MyModel::perturb(DNest4::RNG& rng)
{
    double logH = 0.0;

    // Select blocks of parameters
    int choice = rng.rand_int(4);

    if(choice == 0)
    {
        // Perturb a single n or many ns
        if(rng.rand() <= 0.5)
        {
            int which = rng.rand_int(n.size());
            logH -= -0.5*pow(n[which], 2);
            n[which] += rng.randh();
            logH += -0.5*pow(n[which], 2);
        }
        else
        {
            int reps = (int)pow(10.0, 3*rng.rand());
            for(int i=0; i<reps; ++i)
                n[rng.rand_int(n.size())] = rng.randn();
        }

        compute_wide_component();
    }
    else if(choice == 1)
    {
        // Perturb spikes
        logH += spikes.perturb(rng);

        compute_the_spikes(spikes.get_removed().size() == 0);
    }
    else if(choice == 2)
    {
        // Perturb a parameter related to the
        // background or wide component
        int which = rng.rand_int(6);

        if(which == 0)
        {
            background = log(background);
            logH += laplace.perturb(background, rng);
            background = exp(background);
        }
        else if(which == 1)
        {
            amplitude = log(amplitude);
            logH += laplace.perturb(amplitude, rng);
            amplitude = exp(amplitude);

            compute_wide_component();
        }
        else if(which == 2)
        {
            center += data.get_x_range()*rng.rand();
            DNest4::wrap(center, data.get_x_min(), data.get_x_max());

            compute_wide_component();
        }
        else if(which == 3)
        {
            width += data.get_x_range()*rng.rand();
            DNest4::wrap(width, 0.0, data.get_x_range());

            compute_wide_component();
        }
        else if(which == 4)
        {
            beta = log(beta);
            beta += log(1E6)*rng.randh();
            DNest4::wrap(beta, log(1E-6), log(1.0));
            beta = exp(beta);
        }
        else if(which == 5)
        {
            L = log(L);
            L += log(1E2)*rng.randh();
            DNest4::wrap(L, log(1E-2*data.get_x_range()),
                            log(data.get_x_range()));
            L = exp(L);
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

void MyModel::compute_wide_component()
{
    const auto& data_x = data.get_x();

    double tau = 1.0/(width*width);

    // Make the wide gaussian
    double rsq;
    for(size_t i=0; i<wide_component.size(); ++i)
    {
        rsq = pow(data_x[i] - center, 2);
        wide_component[i] = amplitude*exp(-0.5*rsq*tau);
    }

    // Multiply by the free-form component
    double alpha = exp(-1.0/L);
    double s = beta/sqrt(1.0 - alpha*alpha);
    std::vector<double> ar1(n.size());
    ar1[0] = s*n[0];
    for(size_t i=1; i<ar1.size(); ++i)
        ar1[i] = alpha*ar1[i-1] + beta*n[i];
    for(size_t i=0; i<wide_component.size(); ++i)
        wide_component[i] *= exp(ar1[i]);
}

void MyModel::compute_the_spikes(bool update)
{
    const auto& data_x = data.get_x();

    // Zero the curve if this is not an update
    if(!update)
        for(double& t: the_spikes)
            t = 0.0;

    const auto& components = (update)?(spikes.get_added())
                                :(spikes.get_components());

    double rsq, tau, c, a, w;
    for(size_t i=0; i<components.size(); ++i)
    {
        // {center, log_amplitude, width}
        c = components[i][0];
        a = exp(components[i][1]);
        w = components[i][2];
        tau = 1.0/(w*w);

        for(size_t j=0; j<the_spikes.size(); ++j)
        {
            rsq = pow(data_x[j] - c, 2);
            if(rsq*tau < 100.0)
                the_spikes[j] += a*exp(-0.5*rsq*tau);
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
    double model, resid, var;
    for(size_t i=0; i<data_y.size(); ++i)
    {
        model = background + wide_component[i] + the_spikes[i];
        resid = data_y[i] - model;
        var = sigma0*sigma0 + sigma1*model;

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

    double model;
    for(size_t i=0; i<wide_component.size(); ++i)
    {
        model = background + wide_component[i] + the_spikes[i];
        out<<model<<' ';
    }
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
    for(size_t i=0; i<wide_component.size(); ++i)
        s<<"model_curve["<<i<<"], ";

    return s.str();
}

void MyModel::load_data(const char* filename)
{
    data.load(filename);
}

} // namespace Crystals

