#ifndef Crystals_MyConditionalPrior
#define Crystals_MyConditionalPrior

#include "DNest4/code/DNest4.h"

namespace AMORPH
{

class MyConditionalPrior:public DNest4::ConditionalPrior
{
    private:
        // Data-related stuff which will be useful for the priors
        double x_min, x_max, x_range;

        // For amplitudes
        double location_log_amplitude, scale_log_amplitude;

        // For widths
        double location_log_width, scale_log_width;

        // This determines the prior for location_log_width
        bool narrow;

        double perturb_hyperparameters(DNest4::RNG& rng);

    public:
        MyConditionalPrior(double x_min, double x_max, bool narrow);

        void from_prior(DNest4::RNG& rng);

        double log_pdf(const std::vector<double>& vec) const;
        void from_uniform(std::vector<double>& vec) const;
        void to_uniform(std::vector<double>& vec) const;

        void print(std::ostream& out) const;

        /* Static stuff */
    private:
        static const DNest4::Laplace laplace;
        static const int weight_parameter = 1;
};

} // namespace AMORPH

#endif

