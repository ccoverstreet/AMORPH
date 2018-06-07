#include "Config.h"
#include <yaml-cpp/yaml.h>

namespace AMORPH
{

// Static instance
Config Config::global;

void Config::load(const char* filename)
{

    // Load stuff from the YAML file
    YAML::Node config = YAML::LoadFile(filename);

    data_file = config["data_file"].as<std::string>();
    dnest4_options_file = config["dnest4_options_file"].as<std::string>();
    control_points = std::tuple<double, double>();
    std::get<0>(control_points) = config["control_points"][0].as<double>();
    std::get<1>(control_points) = config["control_points"][1].as<double>();
    max_num_narrow_peaks = config["max_num_narrow_peaks"].as<size_t>();
    max_num_wide_peaks = config["max_num_wide_peaks"].as<size_t>();
}



} // namespace AMORPH

