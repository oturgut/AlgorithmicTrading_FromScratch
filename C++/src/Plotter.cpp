#include "Plotter.h"

Plotter::Plotter() {
    // Initialize matplotlibcpp if not already initialized
    if (!plt::isInitialized()) {
        plt::init();
    }
}

void Plotter::addData(const std::string& label, const std::vector<double>& data) {
    dataSeries[label] = data;
}

void Plotter::plot() {
    for (const auto& entry : dataSeries) {
        plt::named_plot(entry.first, entry.second);
    }

    plt::legend();
    plt::show();
}
