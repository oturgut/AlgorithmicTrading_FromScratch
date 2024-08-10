#include "DataDownloader.h"
#include <iostream>
#include <vector>
#include <curl/curl.h>

DataDownloader::DataDownloader(const std::string& apiUrl) : apiUrl(apiUrl) {
    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl = curl_easy_init();
}

DataDownloader::~DataDownloader() {
    if (curl) {
        curl_easy_cleanup(curl);
    }
    curl_global_cleanup();
}

void DataDownloader::SetApiUrl(const std::string& newApiUrl) {
    apiUrl = newApiUrl;
}

void DataDownloader::DownloadData(const std::vector<std::string>& symbols, const std::string& startDate, const std::string& endDate, std::function<void(const std::string&)> callback) {
    if (!curl) {
        std::cerr << "Curl initialization failed." << std::endl;
        return;
    }

    for (const auto& symbol : symbols) {
        std::string url = apiUrl + "?symbol=" + symbol + "&start_date=" + startDate + "&end_date=" + endDate;
        std::cout << url << std::endl;
        
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());

        std::string downloadedData;
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &downloadedData);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, &DataDownloader::WriteCallback);

        CURLcode res = curl_easy_perform(curl);

        if (res != CURLE_OK) {
            std::cerr << "Error during data download for symbol " << symbol << ": " << curl_easy_strerror(res) << std::endl;
        } else {
            callback(downloadedData);
        }
    }
}

size_t DataDownloader::WriteCallback(void* contents, size_t size, size_t nmemb, std::string* output) {
    size_t total_size = size * nmemb;
    output->append(static_cast<char*>(contents), total_size);
    return total_size;
}
