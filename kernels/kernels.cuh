#pragma once

#include "cuda_runtime.h"

namespace kernels
{
    const int QUADTREE_EMPTY = INT_MIN;
    const int QUADTREE_LOCK = INT_MIN + 1;
    const int QUADTREE_SUCCESS = INT_MIN + 2;

    __constant__ __device__ float const_dev_x_min = 0;
    __constant__ __device__ float const_dev_x_max = 0;
    __constant__ __device__ float const_dev_y_min = 0;
    __constant__ __device__ float const_dev_y_max = 0;
    __constant__ __device__ int *allocated_quadcells_counter;

    __device__ inline bool is_body(const int &quadtree_index)
    {
        return quadtree_index < 0;
    }

    __device__ inline int body_index_to_quadtree_value(const int &body_index)
    {
        return ~body_index;
    }

    __device__ inline int quadtree_value_to_body_index(const int &quadtree_index)
    {
        return ~quadtree_index;
    }

    __device__ inline int quadtree_child_local_index(float x, float y, float cell_x_min, float cell_x_max, float cell_y_min, float cell_y_max)
    {
        int next_child_number = 0;

        float midpoint;

        midpoint = (cell_x_max + cell_x_min) / 2;
        next_child_number += (x < midpoint) ? 0 : 1;
        midpoint = (cell_y_max + cell_y_min) / 2;
        next_child_number += (y < midpoint) ? 0 : 2;

        return next_child_number;
    }

    __device__ inline int quadtree_child_local_index_and_modify_cell_limits(
        float x, float y,
        float &cell_x_min, float &cell_x_max,
        float &cell_y_min, float &cell_y_max)
    {
        int next_child_number = 0;

        float midpoint;

        midpoint = (cell_x_max + cell_x_min) / 2;
        if (x < midpoint)
        {
            cell_x_max = midpoint;
        }
        else
        {
            cell_x_min = midpoint;
            next_child_number += 1;
        }

        midpoint = (cell_y_max + cell_y_min) / 2;
        if (y < midpoint)
        {
            cell_y_max = midpoint;
        }
        else
        {
            cell_y_min = midpoint;
            next_child_number += 2;
        }

        return next_child_number;
    }

    template <int SET_SIZE>
    __global__ void apply_velocities_kernel(float *position_x, float *position_y, float *velocity_x, float *velocity_y, float x_min, float x_max, float y_min, float y_max)
    {
        int index = threadIdx.x + blockIdx.x * blockDim.x;

        if (index < SET_SIZE)
        {
            float p_x = position_x[index];
            float p_y = position_y[index];
            float v_x = velocity_x[index];
            float v_y = velocity_y[index];

            if (p_x > x_max)
            {
                v_x = -v_x;
                p_x -= 2 * (p_x - x_max);
            }
            else if (p_x < x_min)
            {
                v_x = -v_x;
                p_x += 2 * (x_min - p_x);
            }

            if (p_y > y_max)
            {
                v_y = -v_y;
                p_y -= 2 * (p_y - y_max);
            }
            else if (p_y < y_min)
            {
                v_y = -v_y;
                p_y += 2 * (y_min - p_y);
            }

            p_x += v_x;
            p_y += v_y;

            position_x[index] = p_x;
            position_y[index] = p_y;
            velocity_x[index] = v_x;
            velocity_y[index] = v_y;
        }
    }

    template <int QUADTREE_NODES_NUMBER>
    __global__ void clean_quadtree_data_kernel(int *quadtree)
    {
        int index = threadIdx.x + blockIdx.x * blockDim.x;

        if (index < QUADTREE_NODES_NUMBER * 4)
        {
            quadtree[index] = QUADTREE_EMPTY;
        }
    }

    template <int SET_SIZE, int QUADTREE_NODES_NUMBER>
    __global__ void build_quadtree_kernel(int *quadtree, float *position_x, float *position_y)
    {
        int cell_index = 0;
        int body_index = threadIdx.x + blockIdx.x * blockDim.x;
        int success = 0;
        int compute_lower_cell = 1;
        if (body_index >= SET_SIZE)
        {
            success = 1;
        }
        float particle_x = success ? 0 : position_x[body_index];
        float particle_y = success ? 0 : position_y[body_index];
        float current_cell_x_min = const_dev_x_min;
        float current_cell_x_max = const_dev_x_max;
        float current_cell_y_min = const_dev_y_min;
        float current_cell_y_max = const_dev_y_max;
        // insert each body into tree:
        // traverse tree until success flag is set
        // if we find -1 - great, insert body and set success flag
        // if we find cell - traverse down in the corresponding cell
        // if we find -2, we wait (so syncthread and while)
        // if we find body, we insert the other body in its corresponding lower cell ('allocate it'), we write down the lower cell, and we go down until we find -2 or -1

        while (1)
        {
            if (compute_lower_cell)
            {
                cell_index += quadtree_child_local_index_and_modify_cell_limits(
                    particle_x, particle_y,
                    current_cell_x_min, current_cell_x_max,
                    current_cell_y_min, current_cell_y_max);
            }

            int current_quadtree_value = quadtree[cell_index];

            if (success)
            {
                compute_lower_cell = 0;
            }
            else if (current_quadtree_value == QUADTREE_LOCK)
            {
                compute_lower_cell = 0;
            }
            else if (current_quadtree_value == QUADTREE_EMPTY)
            {
                if (atomicCAS(&quadtree[cell_index], QUADTREE_EMPTY, body_index_to_quadtree_value(body_index)) == QUADTREE_EMPTY)
                {
                    success = 1;
                    compute_lower_cell = 0;
                    __threadfence();
                }
                else
                {
                    compute_lower_cell = 0;
                }
            }
            else if (is_body(current_quadtree_value))
            {
                if (atomicCAS(&quadtree[cell_index], current_quadtree_value, QUADTREE_LOCK) == current_quadtree_value)
                {
                    int other_particle_index = quadtree_value_to_body_index(current_quadtree_value);

                    int other_next_child_number = quadtree_child_local_index(
                        position_x[other_particle_index], position_y[other_particle_index],
                        current_cell_x_min, current_cell_x_max,
                        current_cell_y_min, current_cell_y_max);

                    int other_new_parent_cell_numer = ((atomicAdd(allocated_quadcells_counter, 1) + 1) << 2);

                    quadtree[other_new_parent_cell_numer + other_next_child_number] = current_quadtree_value;
                    quadtree[cell_index] = other_new_parent_cell_numer;
                    cell_index = other_new_parent_cell_numer;

                    compute_lower_cell = 1;

                    __threadfence();
                }
                else
                {
                    compute_lower_cell = 0;
                }
            }
            else // is cell
            {
                cell_index = current_quadtree_value;
                compute_lower_cell = 1;
            }

            if (__syncthreads_and(success))
            {
                break;
            }
        }

        /*
        while (1)
        {
            int new_cell_index = cell_index;
            int current_quadtree_value = QUADTREE_SUCCESS;
            if (compute_lower_cell)
            {
                int next_child_number = 0;

                float midpoint;

                midpoint = (current_cell_x_max - current_cell_x_min) / 2;
                if (particle_x < midpoint)
                {
                    current_cell_x_max = midpoint;
                }
                else
                {
                    current_cell_y_min = midpoint;
                    next_child_number += 1;
                }

                midpoint = (current_cell_y_max - current_cell_y_min) / 2;
                if (particle_y < midpoint)
                {
                    current_cell_y_max = midpoint;
                }
                else
                {
                    current_cell_y_min = midpoint;
                    next_child_number += 2;
                }

                new_cell_index += next_child_number;
                current_quadtree_value = quadtree[new_cell_index];
            }

            if (current_quadtree_value == QUADTREE_SUCCESS)
            {
                compute_lower_cell = 0;
            }
            else if (current_quadtree_value == QUADTREE_LOCK)
            {
                compute_lower_cell = 0;
                // do nothing
            }
            else //
            {
                if (current_quadtree_value == QUADTREE_EMPTY)

                {
                    // insert atomically body into quadtree
                    if (atomicCAS(&quadtree[new_cell_index], QUADTREE_EMPTY, body_index_to_quadtree_value(body_index)) == QUADTREE_EMPTY)
                    {
                        success = 1;
                        compute_lower_cell = 0;
                        __threadfence();
                    }
                }
                else if (is_body(current_quadtree_value))
                {
                    if (atomicCAS(&quadtree[new_cell_index], current_quadtree_value, QUADTREE_LOCK) == current_quadtree_value)
                    {
                        int other_particle_index = quadtree_value_to_body_index(current_quadtree_value);
                        float other_particle_x = position_x[other_particle_index];
                        float other_particle_y = position_y[other_particle_index];

                        int other_next_child_number = 0;
                        other_next_child_number += (other_particle_x < (current_cell_x_max - current_cell_x_min) / 2) ? 0 : 1;
                        other_next_child_number += (other_particle_y < (current_cell_y_max - current_cell_y_min) / 2) ? 0 : 2;

                        int other_particle_new_cell_numer = new_cell_index << 2 + other_next_child_number;
                        quadtree[other_particle_new_cell_numer] = current_quadtree_value;
                        quadtree[new_cell_index] = new_cell_index << 2;

                        compute_lower_cell = 1;

                        __threadfence();
                    }
                }
                else // is cell
                {
                    cell_index = current_quadtree_value;
                    compute_lower_cell = 1;
                }
            }

            if (__syncthreads_and(success))
            {
                break;
            }
        }
        */

        return;
    }
}