#pragma once

#include "../particles_set/particles_set.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <stdio.h>

#define cuda_try_or_return(result)                                                               \
    if (result != cudaSuccess)                                                                   \
    {                                                                                            \
        fprintf(stderr, "%s: %d - CUDA action failed! Code: %d.\n", __FILE__, __LINE__, result); \
        return result;                                                                           \
    }

#define GET_VARIABLE_NAME(Variable) (#Variable)

namespace particles
{
    template <int SET_SIZE>
    __global__ void applyVelocitiesKernel(float *position_x, float *position_y, float *velocity_x, float *velocity_y);

    __constant__ __device__ float *const_dev_x_min = 0;
    __constant__ __device__ float *const_dev_x_max = 0;
    __constant__ __device__ float *const_dev_y_min = 0;
    __constant__ __device__ float *const_dev_y_max = 0;

    template <int SET_SIZE>
    class engine
    {
    public:
        static const int THREADS_PER_BLOCK = 1024;
        static engine<SET_SIZE> &instance()
        {
            return _instance;
        }

    private:
        static inline engine<SET_SIZE> _instance = engine();
        float *dev_position_x = nullptr;
        float *dev_position_y = nullptr;
        float *dev_velocity_x = nullptr;
        float *dev_velocity_y = nullptr;
        float *dev_charge = nullptr;
        float *dev_mass = nullptr;

    private:
        engine()
        {
        }

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
            cuda_try_or_return(cudaMalloc((void **)&dev_position_x, SET_SIZE * sizeof(float)));
            cuda_try_or_return(cudaMalloc((void **)&dev_position_y, SET_SIZE * sizeof(float)));
            cuda_try_or_return(cudaMalloc((void **)&dev_velocity_x, SET_SIZE * sizeof(float)));
            cuda_try_or_return(cudaMalloc((void **)&dev_velocity_y, SET_SIZE * sizeof(float)));
            cuda_try_or_return(cudaMalloc((void **)&dev_charge, SET_SIZE * sizeof(float)));
            cuda_try_or_return(cudaMalloc((void **)&dev_mass, SET_SIZE * sizeof(float)));
            cuda_try_or_return(cudaDeviceSynchronize());
            return cudaSetDevice(0);
        }

        cudaError_t load_data_to_gpu(particles_set<SET_SIZE> *particles_set)
        {
            cuda_try_or_return(cudaMemcpy(dev_position_x, particles_set->position_x, SET_SIZE * sizeof(float), cudaMemcpyHostToDevice));
            cuda_try_or_return(cudaMemcpy(dev_position_y, particles_set->position_y, SET_SIZE * sizeof(float), cudaMemcpyHostToDevice));
            cuda_try_or_return(cudaMemcpy(dev_velocity_x, particles_set->velocity_x, SET_SIZE * sizeof(float), cudaMemcpyHostToDevice));
            cuda_try_or_return(cudaMemcpy(dev_velocity_y, particles_set->velocity_y, SET_SIZE * sizeof(float), cudaMemcpyHostToDevice));
            cuda_try_or_return(cudaMemcpy(dev_charge, particles_set->charge, SET_SIZE * sizeof(float), cudaMemcpyHostToDevice));
            cuda_try_or_return(cudaMemcpy(dev_mass, particles_set->mass, SET_SIZE * sizeof(float), cudaMemcpyHostToDevice));

            cuda_try_or_return(cudaDeviceSynchronize());

            return cudaSuccess;
        }

        cudaError_t load_data_from_gpu(particles_set<SET_SIZE> *particles_set)
        {
            cuda_try_or_return(cudaMemcpy(particles_set->position_x, dev_position_x, SET_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
            cuda_try_or_return(cudaMemcpy(particles_set->position_y, dev_position_y, SET_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

            cuda_try_or_return(cudaDeviceSynchronize());

            return cudaSuccess;
        }

        cudaError_t move()
        {
            cuda_try_or_return(set_particles_bounds());
            particles::applyVelocitiesKernel<SET_SIZE><<<SET_SIZE / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK>>>(dev_position_x, dev_position_y, dev_velocity_x, dev_velocity_y);
            cuda_try_or_return(cudaGetLastError());
            return cudaDeviceSynchronize();
        }

        void perform_step()
        {
        }

        /// @brief Sets const_dev_x_min, const_dev_x_max, const_dev_y_min, const_dev_y_max, that are stored in constant memory of GPU.
        /// @return Value of cudaError_t from first unsuccessfull CUDA call.
        cudaError_t set_particles_bounds()
        {
            thrust::device_ptr<float> thrust_position_x = thrust::device_pointer_cast(dev_position_x);
            thrust::device_ptr<float> thrust_position_y = thrust::device_pointer_cast(dev_position_y);

            auto width_limits = thrust::minmax_element(thrust_position_x, thrust_position_x + SET_SIZE);
            auto height_limits = thrust::minmax_element(thrust_position_y, thrust_position_y + SET_SIZE);

            cuda_try_or_return(cudaMemcpyToSymbol(const_dev_x_min, width_limits.first.get(), sizeof(float), 0, cudaMemcpyDeviceToDevice));
            cuda_try_or_return(cudaMemcpyToSymbol(const_dev_x_max, width_limits.second.get(), sizeof(float), 0, cudaMemcpyDeviceToDevice));
            cuda_try_or_return(cudaMemcpyToSymbol(const_dev_y_min, height_limits.first.get(), sizeof(float), 0, cudaMemcpyDeviceToDevice));
            cuda_try_or_return(cudaMemcpyToSymbol(const_dev_y_max, height_limits.second.get(), sizeof(float), 0, cudaMemcpyDeviceToDevice));

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

    template <int SET_SIZE>
    __global__ void calculateBounds(float *position_x, float *position_y)
    {
    }
}