#include "DataDownloader.h"
#include <iostream>
#include <vector>

int main() {
    DataDownloader downloader("https://query1.finance.yahoo.com/v7/finance/download");
    
    std::vector<std::string> symbols = {"AAPL"};
    
    std::string startDate = "2023-01-01";
    std::string endDate = "2023-11-10";

    // Define callback function to process downloaded data
    auto processDownloadedData = [](const std::string& data) {
        // Parse the data and process as needed
        std::cout << "Downloaded data:\n" << data << "\n";
    };
    
    // downloader.SetApiUrl("https://query1.finance.yahoo.com/v7/finance/download");
    downloader.DownloadData(symbols, startDate, endDate, processDownloadedData);
    return 0;
}