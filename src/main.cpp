#include <iostream>
#include "DNest4/code/DNest4.h"

// From this project
#include "Data.h"
#include "MyModel.h"

int main(int argc, char** argv)
{
    // Process command line options
    DNest4::CommandLineOptions clo(argc, argv);

    // Get specified data file. If none, use a default.
    std::string data_file = clo.get_data_file();
    if(data_file.length() == 0)
        data_file = std::string("50% glass .02step3secdwell.txt");

    // Load data
    Crystals::MyModel::load_data(data_file.c_str());

    // Run DNest4.
    DNest4::start<Crystals::MyModel>(clo);

    return 0;
}

