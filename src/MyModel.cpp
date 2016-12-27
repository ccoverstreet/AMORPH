#include "MyModel.h"
#include "DNest4/code/DNest4.h"
#include "Lookup.h"
#include <sstream>
#include <cmath>

namespace Crystals
{

Data MyModel::data;
const DNest4::Laplace MyModel::laplace(0.0, 5.0);

MyModel::MyModel()
:narrow_gaussians(3, max_num_spikes, false,
            MyConditionalPrior(data.get_x_min(), data.get_x_max(), true),
                                DNest4::PriorType::log_uniform)
,wide_gaussians(3, max_num_spikes, false,
            MyConditionalPrior(data.get_x_min(), data.get_x_max(), false),
                                DNest4::PriorType::log_uniform)
,narrow(data.get_y().size())
,wide(data.get_y().size())
{
    if(!data.get_loaded())
        std::cerr<<"# WARNING: it appears no data has been loaded."<<std::endl;
}

void MyModel::from_prior(DNest4::RNG& rng)
{
    background = exp(laplace.generate(rng));

    narrow_gaussians.from_prior(rng);
    wide_gaussians.from_prior(rng);

    compute_narrow();
    compute_wide();

    sigma0 = exp(laplace.generate(rng));
    sigma1 = exp(laplace.generate(rng));
    nu = exp(log(1.0) + log(1E3)*rng.rand());
}

double MyModel::perturb(DNest4::RNG& rng)
{
    double logH = 0.0;

    // Select blocks of parameters
    int choice = rng.rand_int(3);

    if(choice == 0)
    {
        // Perturb spikes
        logH += narrow_gaussians.perturb(rng);

        compute_narrow(narrow_gaussians.get_removed().size() == 0);
    }
    else if(choice == 1)
    {
        // Perturb spikes
        logH += wide_gaussians.perturb(rng);

        compute_wide(wide_gaussians.get_removed().size() == 0);
    }
    else
    {
        // Perturb one of these parameters
        int which = rng.rand_int(4);

        if(which == 0)
        {
            background = log(background);
            logH += laplace.perturb(background, rng);
            background = exp(background);
        }
        else if(which == 1)
        {
            sigma0 = log(sigma0);
            logH += laplace.perturb(sigma0, rng);
            sigma0 = exp(sigma0);
        }
        else if(which == 2)
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


void MyModel::compute_narrow(bool update)
{
    const auto& data_x = data.get_x();

    // Zero the curve if this is not an update
    if(!update)
        for(double& y: narrow)
            y = 0.0;

    const auto& components = (update)?(narrow_gaussians.get_added())
                                :(narrow_gaussians.get_components());

    double rsq, tau, c, a, w;
    for(size_t i=0; i<components.size(); ++i)
    {
        // {center, log_amplitude, width}
        c = components[i][0];
        a = exp(components[i][1]);
        w = components[i][2];
        tau = 1.0/(w*w);

        for(size_t j=0; j<narrow.size(); ++j)
        {
            rsq = pow(data_x[j] - c, 2);
            if(rsq*tau < 100.0)
                narrow[j] += a*Lookup::evaluate(0.5*rsq*tau);
        }
    }
}

void MyModel::compute_wide(bool update)
{
    const auto& data_x = data.get_x();

    // Zero the curve if this is not an update
    if(!update)
        for(double& y: wide)
            y = 0.0;

    const auto& components = (update)?(wide_gaussians.get_added())
                                :(wide_gaussians.get_components());

    double rsq, tau, c, a, w;
    for(size_t i=0; i<components.size(); ++i)
    {
        // {center, log_amplitude, width}
        c = components[i][0];
        a = exp(components[i][1]);
        w = components[i][2];
        tau = 1.0/(w*w);

        for(size_t j=0; j<wide.size(); ++j)
        {
            rsq = pow(data_x[j] - c, 2);
            if(rsq*tau < 100.0)
                wide[j] += a*Lookup::evaluate(0.5*rsq*tau);
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
        model = background + narrow[i] + wide[i];
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
    out<<background<<' ';
    narrow_gaussians.print(out);
    wide_gaussians.print(out);
    out<<sigma0<<' '<<sigma1<<' '<<nu<<' ';


    for(size_t i=0; i<narrow.size(); ++i)
        out<<narrow[i]<<' ';

    for(size_t i=0; i<wide.size(); ++i)
        out<<wide[i]<<' ';

    double model;
    for(size_t i=0; i<wide.size(); ++i)
    {
        model = background + narrow[i] + wide[i];
        out<<model<<' ';
    }
}

std::string MyModel::description() const
{
    std::stringstream s;
    s<<"background, ";
    s<<"dim_gaussians1, max_num_gaussians1, ";
    s<<"location_log_amplitude1, scale_log_amplitude1, ";
    s<<"location_log_width1, scale_log_width1, num_gaussians1, ";
    for(size_t i=0; i<max_num_spikes; ++i)
        s<<"center1["<<i<<"], ";
    for(size_t i=0; i<max_num_spikes; ++i)
        s<<"log_amplitude1["<<i<<"], ";
    for(size_t i=0; i<max_num_spikes; ++i)
        s<<"width1["<<i<<"], ";

    s<<"dim_gaussians2, max_num_gaussians2, ";
    s<<"location_log_amplitude2, scale_log_amplitude2, ";
    s<<"location_log_width2, scale_log_width2, num_gaussians2, ";
    for(size_t i=0; i<max_num_spikes; ++i)
        s<<"center2["<<i<<"], ";
    for(size_t i=0; i<max_num_spikes; ++i)
        s<<"log_amplitude2["<<i<<"], ";
    for(size_t i=0; i<max_num_spikes; ++i)
        s<<"width2["<<i<<"], ";

    s<<"sigma0, sigma1, nu, ";

    for(size_t i=0; i<narrow.size(); ++i)
        s<<"narrow["<<i<<"], ";
    for(size_t i=0; i<wide.size(); ++i)
        s<<"wide["<<i<<"], ";
    for(size_t i=0; i<wide.size(); ++i)
        s<<"model_curve["<<i<<"], ";

    return s.str();
}

void MyModel::load_data(const char* filename)
{
    data.load(filename);
}

} // namespace Crystals

