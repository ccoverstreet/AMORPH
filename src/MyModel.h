#ifndef AMORPH_MyModel_h
#define AMORPH_MyModel_h

#include <ostream>
#include <tuple>
#include "DNest4/code/DNest4.h"
#include "Data.h"
#include "MyConditionalPrior.h"

namespace AMORPH
{

class MyModel
{
    private:
        // Constant background
        double background;

        // Knots in background
        std::vector<double> n_knots;

        // Use an RJObject for the narrow spikes and one for the wide spikes
        DNest4::RJObject<MyConditionalPrior> narrow_peaks, wide_peaks;

        // Shape parameter of narrow peaks
        double peak_shape;

        // Noise-related parameters
        double sigma0, sigma1, nu;

        // Components of the model-predicted curve
        std::vector<double> bg;
        std::vector<double> narrow;
        std::vector<double> wide;

        // Calculate the parts of the model-predicted curve
        void compute_bg();
        void compute_narrow(bool update=false);
        void compute_wide(bool update=false);

    public:
        // Constructor only gives size of params
        MyModel();

        // Generate the point from the prior
        void from_prior(DNest4::RNG& rng);

        // Metropolis-Hastings proposals
        double perturb(DNest4::RNG& rng);

        // Likelihood function
        double log_likelihood() const;

        // Print to stream
        void print(std::ostream& out) const;

        // Return string with column information
        std::string description() const;


        /********** STATIC STUFF **********/
    private:
        // The dataset!
        static Data data;

        // A useful laplace distribution
        static const DNest4::Laplace laplace;

        // x-positions of background knots
        static std::vector<double> x_knots;

    public:
        static void load_data(const char* filename);
        static void set_control_points(const std::tuple<double, double>& cps);
};

} // namespace AMORPH

#endif

