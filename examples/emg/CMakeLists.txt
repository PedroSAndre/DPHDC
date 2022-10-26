add_executable(emg emg.cpp readDataset.cpp readDataset.hpp)
target_link_libraries(emg dphdc readExeInputs ResultsHandler)