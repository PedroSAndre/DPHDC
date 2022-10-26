add_executable(voicehd voicehd.cpp readDataset.cpp readDataset.hpp)
target_link_libraries(voicehd dphdc readExeInputs ResultsHandler)