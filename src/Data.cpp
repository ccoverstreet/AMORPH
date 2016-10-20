#include "Data.h"

#include <fstream>
#include <iostream>
#include <algorithm>

namespace Crystals
{

Data::Data()
:loaded(false)
{

}

void Data::load(const char* filename)
{
    // Open the file
    std::fstream fin(filename, std::ios::in);

    // If the file fails to open
    if(!fin)
    {
        std::cerr<<"# WARNING: Could not open file "<<filename<<"."<<std::endl;
        return;
    }

    // Clear any existing data
    theta.clear();
    count.clear();

    // Temporary variables for reading
    double temp1, temp2;

    // Read the data
    while(fin>>temp1 && fin>>temp2)
    {
        theta.push_back(temp1);
        count.push_back(temp2);
    }

    // Print message to screen
    std::cout<<"# Read "<<theta.size()<<" data points from file ";
    std::cout<<filename<<'.'<<std::endl;

    // Compute summaries
    compute_summaries();

    // Close the file
    fin.close();

    // Set flag
    loaded = true;
}

void Data::compute_summaries()
{
    theta_min = *std::min_element(theta.begin(), theta.end());
    theta_max = *std::max_element(theta.begin(), theta.end());
    theta_range = theta_max - theta_min;
}

} // namespace Crystals

