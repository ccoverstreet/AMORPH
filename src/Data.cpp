#include "Data.h"

#include <fstream>
#include <iostream>
#include <algorithm>

namespace AMORPH
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
    x.clear();
    y.clear();

    // Temporary variables for reading
    double temp1, temp2;

	// Read past comment lines at the top of the file
    while(fin.peek() == '#')
        fin.ignore(1000000, '\n');

    // Read the data
    while(fin>>temp1 && fin>>temp2)
    {
        x.push_back(temp1);
        y.push_back(temp2);
    }

    // Print message to screen
    std::cout<<"# Read "<<x.size()<<" data points from file ";
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
    x_min = *std::min_element(x.begin(), x.end());
    x_max = *std::max_element(x.begin(), x.end());
    x_range = x_max - x_min;
}

} // namespace AMORPH

