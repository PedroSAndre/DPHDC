#ifndef DPHCL_RESULTSHANDLER_H
#define DPHCL_RESULTSHANDLER_H

#include <chrono>
#include <string>

class ResultsHandler {
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_training;
    std::chrono::time_point<std::chrono::high_resolution_clock> finish_training;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_testing;
    std::chrono::time_point<std::chrono::high_resolution_clock> finish_testing;
    std::string version = "v" PROJECT_VERSION_CMAKE;

    std::time_t getEndTime();

public:
    std::string accelerator;
    float success_rate = -1;
    int vector_size = -1;

    void startTraining();

    void stopTraining();

    void startTesting();

    void stopTesting();

    void printToTerminal();

    void printToFile(const std::string &path_to_file);
};


#endif //DPHCL_RESULTSHANDLER_H
