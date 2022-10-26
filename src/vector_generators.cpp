#include "HDMatrix.hpp"
#include <random>

namespace dphdc {
    // Helper function //
    void HDMatrix::copyBoolVector(const bool *vector_to_copy) {
        {
            cl::sycl::buffer<bool, 2> vector_on_host_buff(vector_to_copy, this->vectors_buff.get_range());
            this->associated_queue.submit([&](cl::sycl::handler &h) {
                cl::sycl::accessor acc_vector_device(this->vectors_buff, h, cl::sycl::write_only);
                cl::sycl::accessor acc_vector_host(vector_on_host_buff, h, cl::sycl::read_only);
                h.parallel_for(this->vectors_buff.get_range(), [=](cl::sycl::id<2> i) {
                    acc_vector_device[i[0]][i[1]] = acc_vector_host[i[0]][i[1]];
                });
            });
        }
    }

    void HDMatrix::constantVectorGenerator(bool value) {
        this->associated_queue.submit([&](cl::sycl::handler &h) {
            cl::sycl::accessor acc(this->vectors_buff, h, cl::sycl::write_only);
            h.parallel_for(this->vectors_buff.get_range(), [=](cl::sycl::id<2> i) {
                acc[i[0]][i[1]] = value;
            });
        });
    }

    void HDMatrix::randomVectorGenerator() {
        std::random_device dev;
        std::mt19937 rng(dev());
        std::bernoulli_distribution dist;

        cl::sycl::range<2> buffer_range = this->vectors_buff.get_range();
        std::unique_ptr<bool[]> vector_on_host(new bool[buffer_range[0] * buffer_range[1]]);


        for (unsigned int i = 0; i < buffer_range[0] * buffer_range[1]; i++) {
            vector_on_host.get()[i] = dist(rng);
        }

        this->copyBoolVector(vector_on_host.get());
    }

    void HDMatrix::levelVectorGenerator(bool full_level) {
        unsigned int vector_size = this->vectors_buff.get_range()[1];
        std::random_device rd; // obtain a random number from hardware
        std::mt19937 gen(rd()); // seed the generator
        std::uniform_int_distribution<unsigned int> dist_int(0, vector_size - 1); // define the range
        std::bernoulli_distribution dist_bern; // define the range

        unsigned int n_bits_to_shift = vector_size;
        if (!full_level) {
            n_bits_to_shift /= 2;
        }
        std::vector<unsigned int> bits_to_shift(n_bits_to_shift);
        if (full_level) {
            for (size_t i = 0; i < bits_to_shift.size(); i++) {
                bits_to_shift[i] = i;
            }
        } else {
            unsigned int random_position;
            bool inserted;
            for (unsigned int i = 0; i < n_bits_to_shift; i++) {
                inserted = false;
                while (!inserted) {
                    random_position = dist_int(gen);
                    auto iterator = std::find(bits_to_shift.begin(), bits_to_shift.end(), random_position);
                    if (iterator == bits_to_shift.end()) {
                        inserted = true;
                        bits_to_shift[i] = random_position;
                    }
                }
            }
        }

        std::unique_ptr<bool[]> vector_to_copy(
                new bool[this->vectors_buff.get_range()[0] * this->vectors_buff.get_range()[1]]);

        for (unsigned int i = 0; i < vector_size; i++) {
            vector_to_copy.get()[i] = dist_bern(gen);
        };

        unsigned int n_bits_to_shift_iteration = n_bits_to_shift / (this->vectors_buff.get_range()[0] - 1);
        unsigned int j_previous;
        unsigned int j_this = 0;
        for (unsigned int i = 1; i < this->vectors_buff.get_range()[0]; i++) {
            j_previous = j_this;
            j_this += vector_size;

            for (unsigned int j = 0; j < vector_size; j++) {
                vector_to_copy.get()[j_this + j] = vector_to_copy.get()[j_previous + j];
            }

            for (unsigned int j = 0; j < n_bits_to_shift_iteration; j++) {
                vector_to_copy.get()[j_this + bits_to_shift.back()] = !vector_to_copy.get()[j_this +
                                                                                            bits_to_shift.back()];
                bits_to_shift.pop_back();
            }
        }

        this->copyBoolVector(vector_to_copy.get());
    }


    void HDMatrix::circularVectorGenerator() {
        size_t n_vectors = this->vectors_buff.get_range()[0];
        size_t vector_size = this->vectors_buff.get_range()[1];
        std::random_device rd; // obtain a random number from hardware
        std::mt19937 gen(rd()); // seed the generator
        std::bernoulli_distribution dist_bern; // define the range


        std::vector<size_t> bits_to_shift(vector_size);
        for (size_t i = 0; i < bits_to_shift.size(); i++) {
            bits_to_shift[i] = i;
        }

        std::unique_ptr<bool[]> vector_to_copy(
                new bool[n_vectors * vector_size]);

        for (size_t i = 0; i < vector_size; i++) {
            vector_to_copy.get()[i] = dist_bern(gen);
            vector_to_copy.get()[(n_vectors / 2) * vector_size + i] = !vector_to_copy.get()[i];
        };

        size_t n_bits_to_shift_iteration = (vector_size) / ((n_vectors / 2));
        size_t j_previous;
        size_t j_this = 0;
        for (size_t i = 1; i < n_vectors / 2; i++) {
            j_previous = j_this;
            j_this += vector_size;

            for (size_t j = 0; j < vector_size; j++) {
                vector_to_copy.get()[j_this + j] = vector_to_copy.get()[j_previous + j];
            }

            for (size_t j = 0; j < n_bits_to_shift_iteration; j++) {
                vector_to_copy.get()[j_this + bits_to_shift.back()] = !vector_to_copy.get()[j_this +
                                                                                            bits_to_shift.back()];
                bits_to_shift.pop_back();
            }

            for (size_t j = 0; j < vector_size; j++) {
                vector_to_copy.get()[j_this + (n_vectors / 2) * vector_size + j] = !vector_to_copy.get()[j_this + j];
            }
        }
        if (n_vectors % 2 != 0) {
            for (size_t i = 0; i < vector_size; i++) {
                if (i < n_bits_to_shift_iteration / 2 || i >= (vector_size - n_bits_to_shift_iteration / 2)) {
                    vector_to_copy.get()[(n_vectors - 1) * vector_size + i] = !vector_to_copy.get()[
                            (n_vectors - 2) * vector_size + i];
                } else {
                    vector_to_copy.get()[(n_vectors - 1) * vector_size + i] = vector_to_copy.get()[
                            (n_vectors - 2) * vector_size + i];
                }
            }
        }


        this->copyBoolVector(vector_to_copy.get());


    }
}