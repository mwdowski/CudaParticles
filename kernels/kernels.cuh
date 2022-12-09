#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace kernels
{
    const int QUADTREE_EMPTY = -1;
    const int QUADTREE_LOCK = -2;
    const int MIN_BODY_INDEX = INT_MAX >> 1;

    __constant__ __device__ float *const_dev_x_min = 0;
    __constant__ __device__ float *const_dev_x_max = 0;
    __constant__ __device__ float *const_dev_y_min = 0;
    __constant__ __device__ float *const_dev_y_max = 0;

    __device__ inline bool is_body(const int &quadtree_index)
    {
        return quadtree_index >= MIN_BODY_INDEX;
    }

    __device__ inline int quadtree_body(const int &body_index)
    {
        return body_index + MIN_BODY_INDEX;
    }

    /// @brief Compute Barnes-Hut quadtree's child index. Quadtree is represented as one array.
    /// @param parent_index Index of parent node. Root note has index 0.
    /// @param child_number Number of child node. Value must be between 0 and 3 (inclusive).
    /// @return Index of child node.
    __device__ inline int quadtree_child_index(int parent_index, int child_number)
    {
        return (parent_index << 2) + child_number + 1;
    }

    template <int QUADTREE_NODES_NUMBER>
    __device__ inline int find_insertion_point(const float &particle_x, const float &particle_y, const int *quadtree)
    {
        int insertion_index = 0;
        int child_number = 0;

        float cell_x_min = *const_dev_x_min;
        float cell_y_min = *const_dev_y_min;
        float cell_x_max = *const_dev_x_max;
        float cell_y_max = *const_dev_y_max;

        while (insertion_index < (QUADTREE_NODES_NUMBER >> 2))
        {
            float cell_x_middle = (cell_x_min + cell_x_max) / 2;
            float cell_y_middle = (cell_y_min + cell_y_max) / 2;

            if (particle_x > cell_x_middle)
            {
                child_number += 1;
                cell_x_min = cell_x_middle;
            }
            else
            {
                cell_y_max = cell_x_middle;
            }

            if (particle_y > cell_y_middle)
            {
                child_number += 2;
                cell_y_min = cell_y_middle;
            }
            else
            {
                cell_y_max = cell_y_middle;
            }

            insertion_index = quadtree_child_index(insertion_index, child_number);
        }

        return insertion_index;
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
    __global__ void clean_quadtree_data_kernel(int *quadtree_body_index)
    {
        int index = threadIdx.x + blockIdx.x * blockDim.x;

        if (index < QUADTREE_NODES_NUMBER * 4)
        {
            quadtree_body_index[index] = QUADTREE_EMPTY;
        }
    }

    template <int SET_SIZE, int QUADTREE_NODES_NUMBER>
    __global__ void build_quadtree_kernel(int *quadtree, float *position_x, float *position_y)
    {
        int cell_index = 0;

        // insert each body into tree:
        // traverse tree until success flag is set
        // if we find -1 - great, insert body and set success flag
        // if we find cell - traverse down in the corresponding cell
        // if we find -2, we wait (so syncthread and while)
        // if we find body, we insert the other body in its corresponding lower cell ('allocate it'), we write down the lower cell, and we go down until we find -2 or -1
        return;
    }
}