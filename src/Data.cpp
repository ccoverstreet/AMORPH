#include "Data.h"
#include <fstream>
#include <iostream>

namespace Crystals
{

Data::Data()
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


    fin.close();
}

} // namespace Crystals

