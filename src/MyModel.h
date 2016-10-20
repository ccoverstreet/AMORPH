#ifndef Crystals_MyModel
#define Crystals_MyModel

#include <ostream>
#include "DNest4/code/DNest4.h"
#include "Data.h"
#include "MyConditionalPrior.h"

namespace Crystals
{

class MyModel
{
    private:
        // Constant background
        double background;

        /* Static variables */
        // The dataset!
        static Data data;

        // A useful Cauchy distribution
        static const DNest4::Cauchy cauchy;

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

        /* Static functions */
        static void load_data(const char* filename);
};

} // namespace Crystals

#endif
