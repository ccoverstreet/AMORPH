#ifndef AMORPH_Config_h
#define AMORPH_Config_h

#include <string>
#include <tuple>

namespace AMORPH
{

class Config
{
    private:

        std::string data_file;
        std::string dnest4_options_file;

        std::tuple<double, double> control_points;

        size_t max_num_narrow_peaks;
        size_t max_num_wide_peaks;

    public:

        // Load from YAML file
        void load(const char* filename);

        // A global instance
        static Config global;

};

} // namespace AMORPH

#endif

