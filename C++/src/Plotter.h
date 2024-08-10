#ifndef PLOTTER_H
#define PLOTTER_H

#include "/Users/ozhanturgut/Documents/GitHub/matplotlib-cpp/matplotlibcpp.h"
#include <vector>
#include <map>
#include <string>

namespace plt = matplotlibcpp;

class Plotter {
public:
    Plotter();

    // Add a data series to the plot
    void addData(const std::string& label, const std::vector<double>& data);

    // Plot the added data
    void plot();

private:
    std::map<std::string, std::vector<double>> dataSeries;
};

#endif // PLOTTER_H
