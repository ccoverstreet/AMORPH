#include <iostream>
#include "DNest4/code/DNest4.h"

// From this project
#include "Data.h"
#include "MyModel.h"

int main(int argc, char** argv)
{
    // Process command line options
    DNest4::CommandLineOptions clo(argc, argv);

    // Load data
    Crystals::MyModel::load_data(clo.get_data_file().c_str());

    // Run DNest4.
    DNest4::start<Crystals::MyModel>(clo);

    return 0;
}

