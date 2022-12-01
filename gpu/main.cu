#include <stdio.h>
#include "../application/application.cuh"

#define PARTICLES_NUMBER 10000

#define cuda_try_or_exit(result)                                               \
    if (result != cudaSuccess)                                                 \
    {                                                                          \
        fprintf(stderr, "%s: %d - CUDA action failed!\n", __FILE__, __LINE__); \
        return 1;                                                              \
    }

int main(int argc, char *argv[])
{
    application::start(argc, argv);
    return 0;
}