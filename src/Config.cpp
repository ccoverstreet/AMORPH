#include "Config.h"
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <yaml-cpp/yaml.h>

namespace AMORPH
{

// Static instance
Config Config::global;

Config::Config()
:argc(0)
,argv(nullptr)
{

}

Config::~Config()
{
    if(argc > 0)
    {
        for(int i=0; i<argc; ++i)
            delete[] argv[i];
        delete[] argv;
    }

}

void Config::load(const char* filename)
{
    if(argc != 0)
    {
        std::cerr << "Don't load config twice." << std::endl;
        return;
    }

    // Load stuff from the YAML file
    YAML::Node config = YAML::LoadFile(filename);

    num_threads = config["num_threads"].as<size_t>();
    data_file = config["data_file"].as<std::string>();
    dnest4_options_file = config["dnest4_options_file"].as<std::string>();
    control_points = std::tuple<double, double>();
    std::get<0>(control_points) = config["inference_assumptions"]
                                        ["control_points"][0].as<double>();
    std::get<1>(control_points) = config["inference_assumptions"]
                                        ["control_points"][1].as<double>();
    max_num_narrow_peaks = config["inference_assumptions"]
                                 ["max_num_narrow_peaks"].as<size_t>();
    max_num_wide_peaks = config["inference_assumptions"]
                               ["max_num_wide_peaks"].as<size_t>();

    argc = 7;
    argv = new char*[argc];
    for(int i=0; i<argc; ++i)
        argv[i] = new char[1000];
    strcpy(argv[0], "main");
    strcpy(argv[1], "-t");
    std::stringstream ss;
    ss << num_threads;
    strcpy(argv[2], ss.str().c_str());
    strcpy(argv[3], "-d");
    strcpy(argv[4], data_file.c_str());
    strcpy(argv[5], "-o");
    strcpy(argv[6], dnest4_options_file.c_str());
}



} // namespace AMORPH

