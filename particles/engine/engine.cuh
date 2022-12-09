#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../particles_set/particles_set.hpp"
#include "../../macros/macros.cuh"
#include "../../kernels/kernels.cuh"

#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <stdio.h>

namespace particles
{
    const int THREADS_PER_BLOCK = 1024;

    template <int SET_SIZE>
    class engine
    {
    public:
        static engine<SET_SIZE> &instance()
        {
            return _instance;
        }

    private:
        static const int QUADTREE_NODES_NUMBER = SET_SIZE * 12;
        static inline engine<SET_SIZE> _instance = engine();
        float *dev_position_x = nullptr;
        float *dev_position_y = nullptr;
        float *dev_velocity_x = nullptr;
        float *dev_velocity_y = nullptr;
        float *dev_charge = nullptr;
        float *dev_mass = nullptr;

        int *dev_quadtree = nullptr;

    private:
        engine()
        {
        }

        /// @brief Free device allocated global memory.
        void clear_data()
        {
            cudaFree(dev_position_x);
            cudaFree(dev_position_y);
            cudaFree(dev_velocity_x);
            cudaFree(dev_velocity_y);
            cudaFree(dev_charge);
            cudaFree(dev_mass);
            cudaFree(dev_quadtree);
        }

    public:
        ~engine()
        {
            clear_data();
        }

        /// @brief Initiate particles engine by setting GPU device and allocating device global memory.
        /// @return Value of cudaError_t from first unsuccessfull CUDA call.
        cudaError_t initiate()
        {
            cuda_try_or_return(cudaSetDevice(0));

            cuda_try_or_return(cudaMalloc((void **)&dev_position_x, SET_SIZE * sizeof(float)));
            cuda_try_or_return(cudaMalloc((void **)&dev_position_y, SET_SIZE * sizeof(float)));
            cuda_try_or_return(cudaMalloc((void **)&dev_velocity_x, SET_SIZE * sizeof(float)));
            cuda_try_or_return(cudaMalloc((void **)&dev_velocity_y, SET_SIZE * sizeof(float)));
            cuda_try_or_return(cudaMalloc((void **)&dev_charge, SET_SIZE * sizeof(float)));
            cuda_try_or_return(cudaMalloc((void **)&dev_mass, SET_SIZE * sizeof(float)));

            cuda_try_or_return(cudaMalloc((void **)&dev_quadtree, QUADTREE_NODES_NUMBER * 4 * sizeof(int)));

            return cudaDeviceSynchronize();
        }

        /// @brief Copy host particles data to device global memory.
        /// @param particles_set Pointer to previously created particles set.
        /// @return Value of cudaError_t from first unsuccessfull CUDA call.
        cudaError_t load_data_to_gpu(particles_set<SET_SIZE> *particles_set)
        {
            cuda_try_or_return(cudaMemcpy(dev_position_x, particles_set->position_x, SET_SIZE * sizeof(float), cudaMemcpyHostToDevice));
            cuda_try_or_return(cudaMemcpy(dev_position_y, particles_set->position_y, SET_SIZE * sizeof(float), cudaMemcpyHostToDevice));
            cuda_try_or_return(cudaMemcpy(dev_velocity_x, particles_set->velocity_x, SET_SIZE * sizeof(float), cudaMemcpyHostToDevice));
            cuda_try_or_return(cudaMemcpy(dev_velocity_y, particles_set->velocity_y, SET_SIZE * sizeof(float), cudaMemcpyHostToDevice));
            cuda_try_or_return(cudaMemcpy(dev_charge, particles_set->charge, SET_SIZE * sizeof(float), cudaMemcpyHostToDevice));
            cuda_try_or_return(cudaMemcpy(dev_mass, particles_set->mass, SET_SIZE * sizeof(float), cudaMemcpyHostToDevice));

            return cudaDeviceSynchronize();
        }

        cudaError_t load_data_from_gpu(particles_set<SET_SIZE> *particles_set)
        {
            cuda_try_or_return(cudaMemcpy(particles_set->position_x, dev_position_x, SET_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
            cuda_try_or_return(cudaMemcpy(particles_set->position_y, dev_position_y, SET_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

            return cudaDeviceSynchronize();
        }

        cudaError_t move(float x_min, float x_max, float y_min, float y_max)
        {
            cuda_try_or_return(set_particles_bounds());

            kernels::apply_velocities_kernel<SET_SIZE><<<SET_SIZE / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK>>>(
                dev_position_x,
                dev_position_y,
                dev_velocity_x,
                dev_velocity_y,
                x_min,
                x_max,
                y_min,
                y_max);
            cuda_try_or_return(cudaDeviceSynchronize());
            cuda_try_or_return(cudaGetLastError());

            kernels::clean_quadtree_data_kernel<QUADTREE_NODES_NUMBER><<<SET_SIZE / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK>>>(dev_quadtree);
            cuda_try_or_return(cudaDeviceSynchronize());
            cuda_try_or_return(cudaGetLastError());

            return cudaDeviceSynchronize();
        }

        void perform_step()
        {
        }

        /// @brief Set const_dev_x_min, const_dev_x_max, const_dev_y_min, const_dev_y_max, that are stored in constant memory of GPU.
        /// @return Value of cudaError_t from first unsuccessfull CUDA call.
        cudaError_t set_particles_bounds()
        {
            // create thrust pointers to device global variables
            thrust::device_ptr<float> thrust_position_x = thrust::device_pointer_cast(dev_position_x);
            thrust::device_ptr<float> thrust_position_y = thrust::device_pointer_cast(dev_position_y);

            // perform thrust minmax operation
            auto width_limits = thrust::minmax_element(thrust_position_x, thrust_position_x + SET_SIZE);
            auto height_limits = thrust::minmax_element(thrust_position_y, thrust_position_y + SET_SIZE);

            // copy results into device constant memory
            cuda_try_or_return(cudaMemcpyToSymbol(kernels::const_dev_x_min, width_limits.first.get(), sizeof(float), 0, cudaMemcpyDeviceToDevice));
            cuda_try_or_return(cudaMemcpyToSymbol(kernels::const_dev_x_max, width_limits.second.get(), sizeof(float), 0, cudaMemcpyDeviceToDevice));
            cuda_try_or_return(cudaMemcpyToSymbol(kernels::const_dev_y_min, height_limits.first.get(), sizeof(float), 0, cudaMemcpyDeviceToDevice));
            cuda_try_or_return(cudaMemcpyToSymbol(kernels::const_dev_y_max, height_limits.second.get(), sizeof(float), 0, cudaMemcpyDeviceToDevice));

            return cudaDeviceSynchronize();
        }
    };

}