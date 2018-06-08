#include <iostream>
#include <fstream>
#include "DNest4/code/DNest4.h"

// From this project
#include "Config.h"
#include "Data.h"
#include "MyModel.h"

using namespace AMORPH;

int main()
{
    // Load the config file
    Config::global.load("config.yaml");

    // Extract the mock command line options
    DNest4::CommandLineOptions clo(Config::global.get_argc(),
                                   Config::global.get_argv());

    // Set the control points
    AMORPH::MyModel::load_control_points("control_points.in");

    // Load data
    AMORPH::MyModel::load_data(Config::global.get_data_file().c_str());

//    // Run DNest4.
//    DNest4::start<AMORPH::MyModel>(clo);

    return 0;
}

