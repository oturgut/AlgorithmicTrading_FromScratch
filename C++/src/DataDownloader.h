#ifndef DATA_DOWNLOADER_H
#define DATA_DOWNLOADER_H

#include <string>
#include <vector>
#include <functional>
#include <curl/curl.h>

class DataDownloader {
public:
    // Constructor
    DataDownloader(const std::string& apiUrl);

    // Destructor
    ~DataDownloader();

    // Set the API URL
    void SetApiUrl(const std::string& newApiUrl);

    // Download data for multiple symbols concurrently
    void DownloadData(const std::vector<std::string>& symbols, const std::string& startDate, const std::string& endDate, std::function<void(const std::string&)> callback);

private:
    // Static callback function for writing received data to a string
    static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* output);

    // Member variables
    std::string apiUrl;
    CURL* curl;

};

#endif
