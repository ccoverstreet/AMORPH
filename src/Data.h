#ifndef Crystals_Data
#define Crystals_Data

#include <vector>

namespace Crystals
{

/*
* An object of this class is a dataset, from which we do the inference.
*/
class Data
{
    private:
        std::vector<double> theta;
        std::vector<int> count;

        // Some summaries
        double theta_min, theta_max, theta_range;

    public:
        // Construct as empty dataset
        Data();

        // Load data from a text file
        void load(const char* filename);

        // Getters (by reference - use caution)
        const std::vector<double>& get_theta() const { return theta; }
        const std::vector<int>& get_count() const { return count; }

        // More getters (not by reference)
        double get_theta_min() const { return theta_min; }
        double get_theta_max() const { return theta_max; }
        double get_theta_range() const { return theta_range; }
};

} // namespace Crystals

#endif

