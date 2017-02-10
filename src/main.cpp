#include <iostream>
#include <fstream>
#include "DNest4/code/DNest4.h"

// From this project
#include "Data.h"
#include "MyModel.h"

int main(int argc, char** argv)
{
    // Process command line options
    DNest4::CommandLineOptions clo(argc, argv);

    // Get specified data file. If none, ask the user.
    std::string data_file = clo.get_data_file();
    if(data_file.length() == 0)
    {
        std::cout << "# Enter the name of the data file you want to load: ";
        std::cin >> data_file;
    }

    // Load the control points
    Crystals::MyModel::load_control_points("control_points.in");

    // Save the data filename
    std::fstream fout("run_data.txt", std::ios::out);
    fout<<data_file;
    fout.close();

    // Load data
    Crystals::MyModel::load_data(data_file.c_str());

    // Run DNest4.
    DNest4::start<Crystals::MyModel>(clo);

    return 0;
}

