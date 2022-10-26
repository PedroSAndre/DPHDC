#include "ResultsHandler.hpp"

#include <iostream>
#include <fstream>

void ResultsHandler::startTraining() {
    this->start_training = std::chrono::high_resolution_clock::now();
}

void ResultsHandler::stopTraining() {
    this->finish_training = std::chrono::high_resolution_clock::now();
}

void ResultsHandler::startTesting() {
    this->start_testing = std::chrono::high_resolution_clock::now();
}

void ResultsHandler::stopTesting() {
    this->finish_testing = std::chrono::high_resolution_clock::now();
}

void ResultsHandler::printToTerminal() {
    std::time_t end_time = this->getEndTime();

    std::cout << "\n\n\nResults\n";
    std::cout << "Library Version: " << this->version << "\n";
    std::cout << "Finished at: " << std::ctime(&end_time);
    std::cout << "Accelerator: " << this->accelerator << "\n";
    std::cout << "Success Rate: " << this->success_rate << "%\n";
    std::cout << "Training Time: " << (float) std::chrono::duration_cast<std::chrono::microseconds>(
            this->finish_training - this->start_training).count() / 1000000 << "s\n";
    std::cout << "Testing Time: " << (float) std::chrono::duration_cast<std::chrono::microseconds>(
            this->finish_testing - this->start_testing).count() / 1000000 << "s\n";
    std::cout << "Vector Size: " << this->vector_size << "\n";
    std::cout << "\n\n";
}

void ResultsHandler::printToFile(const std::string &path_to_file) {
    std::ofstream write_file;

    std::time_t end_time = this->getEndTime();
    std::string end_time_string = std::ctime(&end_time);
    end_time_string.pop_back();

    write_file.open(path_to_file, std::ios_base::app);

    write_file << this->version << ",";
    write_file << end_time_string << ",";
    write_file << this->accelerator << ",";
    write_file << this->success_rate << ",";
    write_file << (float) std::chrono::duration_cast<std::chrono::microseconds>(
            this->finish_training - this->start_training).count() / 1000000 << ",";
    write_file << (float) std::chrono::duration_cast<std::chrono::microseconds>(
            this->finish_testing - this->start_testing).count() / 1000000 << ",";
    write_file << this->vector_size << "\n";

    write_file.close();
}

std::time_t ResultsHandler::getEndTime() {
    std::time_t end_time;
    if ((this->finish_testing - this->start_testing).count() == 0) {
        end_time = std::chrono::system_clock::to_time_t(this->finish_training);
    } else {
        end_time = std::chrono::system_clock::to_time_t(this->finish_testing);
    }

    return end_time;
}
