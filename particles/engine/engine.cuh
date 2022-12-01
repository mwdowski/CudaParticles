#pragma once

#include "../particles_set/particles_set.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define cuda_try_or_return(result) \
    if (result != cudaSuccess)     \
        return result;

namespace particles
{
    template <int SET_SIZE>
    __global__ void applyVelocitiesKernel(float *position_x, float *position_y, float *velocity_x, float *velocity_y);

    template <int SET_SIZE>
    class engine
    {
    public:
        static const int THREADS_PER_BLOCK = 1024;

    private:
        float *dev_position_x = nullptr;
        float *dev_position_y = nullptr;
        float *dev_velocity_x = nullptr;
        float *dev_velocity_y = nullptr;
        float *dev_charge = nullptr;
        float *dev_mass = nullptr;

    private:
        void clear_data()
        {
            cudaFree(dev_position_x);
            cudaFree(dev_position_y);
            cudaFree(dev_velocity_x);
            cudaFree(dev_velocity_y);
            cudaFree(dev_charge);
            cudaFree(dev_mass);
        }

    public:
        ~engine()
        {
            clear_data();
        }

        cudaError_t initiate()
        {
            return cudaSetDevice(0);
        }

        cudaError_t load_data_to_gpu(particles_set<SET_SIZE> &particles_set)
        {
            cuda_try_or_return(cudaMalloc((void **)&dev_position_x, SET_SIZE * sizeof(float)));
            cuda_try_or_return(cudaMalloc((void **)&dev_position_y, SET_SIZE * sizeof(float)));
            cuda_try_or_return(cudaMalloc((void **)&dev_velocity_x, SET_SIZE * sizeof(float)));
            cuda_try_or_return(cudaMalloc((void **)&dev_velocity_y, SET_SIZE * sizeof(float)));
            cuda_try_or_return(cudaMalloc((void **)&dev_charge, SET_SIZE * sizeof(float)));
            cuda_try_or_return(cudaMalloc((void **)&dev_mass, SET_SIZE * sizeof(float)));

            cuda_try_or_return(cudaMemcpy(dev_position_x, particles_set.position_x, SET_SIZE * sizeof(float), cudaMemcpyHostToDevice));
            cuda_try_or_return(cudaMemcpy(dev_position_y, particles_set.position_y, SET_SIZE * sizeof(float), cudaMemcpyHostToDevice));
            cuda_try_or_return(cudaMemcpy(dev_velocity_x, particles_set.velocity_x, SET_SIZE * sizeof(float), cudaMemcpyHostToDevice));
            cuda_try_or_return(cudaMemcpy(dev_velocity_y, particles_set.velocity_y, SET_SIZE * sizeof(float), cudaMemcpyHostToDevice));
            cuda_try_or_return(cudaMemcpy(dev_charge, particles_set.charge, SET_SIZE * sizeof(float), cudaMemcpyHostToDevice));
            cuda_try_or_return(cudaMemcpy(dev_mass, particles_set.mass, SET_SIZE * sizeof(float), cudaMemcpyHostToDevice));

            return cudaSuccess;
        }

        cudaError_t load_data_from_gpu(particles_set<SET_SIZE> &particles_set)
        {
            cuda_try_or_return(cudaMemcpy(particles_set.position_x, dev_position_x, SET_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
            cuda_try_or_return(cudaMemcpy(particles_set.position_y, dev_position_y, SET_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

            return cudaSuccess;
        }

        cudaError_t move()
        {
            particles::applyVelocitiesKernel<SET_SIZE><<<SET_SIZE / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK>>>(dev_position_x, dev_position_y, dev_velocity_x, dev_velocity_y);
            cuda_try_or_return(cudaGetLastError());
            return cudaDeviceSynchronize();
        }
    };

    template <int SET_SIZE>
    __global__ void applyVelocitiesKernel(float *position_x, float *position_y, float *velocity_x, float *velocity_y)
    {
        int index = threadIdx.x + blockIdx.x * engine<SET_SIZE>::THREADS_PER_BLOCK;

        if (index < SET_SIZE)
        {
            position_x[index] += velocity_x[index];
            position_y[index] += velocity_y[index];
        }
    }
}