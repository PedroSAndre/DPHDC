add_executable(hdna hdna.cpp readDataset.cpp readDataset.hpp)
target_link_libraries(hdna dphdc readExeInputs ResultsHandler)