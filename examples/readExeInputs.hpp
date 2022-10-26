#ifndef DPHDC_READEXEINPUTS_HPP
#define DPHDC_READEXEINPUTS_HPP

struct Inputs {
    int vector_size = 10000;
    int n_gram = 3;
};

Inputs readExeInputs(int argc, char **argv);

#endif //DPHDC_READEXEINPUTS_HPP
