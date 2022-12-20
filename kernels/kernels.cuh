#pragma once

#include "cuda_runtime.h"

namespace kernels
{
    __constant__ __device__ const int QUADTREE_EMPTY = INT_MIN;
    __constant__ __device__ const int QUADTREE_LOCK = INT_MIN + 1;
    __constant__ __device__ const int QUADTREE_SUCCESS = INT_MIN + 2;
    __constant__ __device__ const int WARP_SIZE = 32;
    __constant__ __device__ const int STACK_SIZE = 64;
    __constant__ __device__ const float THETA = 1.0f;
    __constant__ __device__ const float EPS = 0.00025f;
    __constant__ __device__ const float K = 14.3996f; // eV * Å / e^2
    __constant__ __device__ const float SCALE_MULTIPLIER = 0.0000001f;
    __constant__ __device__ const int THREADS_PER_BLOCK = 512;

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

    template <int SET_SIZE>
    __device__ inline int quadtree_cell_index_to_index_in_physical_arrays(const int &quadtree_index)
    {
        return (quadtree_index >> 2) + SET_SIZE;
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
    __global__ void clean_quadtree_children_data_kernel(int *quadtree)
    {
        int index = threadIdx.x + blockIdx.x * blockDim.x;

        if (index < QUADTREE_NODES_NUMBER * 4)
        {
            quadtree[index] = QUADTREE_EMPTY;
        }
    }

    template <int SET_SIZE, int QUADTREE_NODES_NUMBER>
    __global__ void clean_quadtree_physical_data_kernel(float *position_x, float *position_y, float *charge)
    {
        int index = threadIdx.x + blockIdx.x * blockDim.x;

        position_x += SET_SIZE;
        position_y += SET_SIZE;
        charge += SET_SIZE;

        if (index < QUADTREE_NODES_NUMBER)
        {
            position_x[index] = 0;
            position_y[index] = 0;
            charge[index] = 0;
        }
    }

    template <int SET_SIZE, int QUADTREE_NODES_NUMBER>
    __global__ void build_quadtree_kernel(int *quadtree, float *position_x, float *position_y, float *charge)
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
        float particle_charge = success ? 0 : charge[body_index];
        float current_cell_x_min = const_dev_x_min;
        float current_cell_x_max = const_dev_x_max;
        float current_cell_y_min = const_dev_y_min;
        float current_cell_y_max = const_dev_y_max;

        // insert each body into tree:
        // traverse tree until success flag is set
        // if we find EMPTY - great, insert body and set success flag
        // if we find cell - traverse down in the corresponding cell
        // if we find LOCK, we wait (so syncthread and while)
        // if we find body, we insert the other body in its corresponding lower cell ('allocate it'), we write down the lower cell, and we continue going down

        while (1)
        {
            if (success)
            {
                // do nothing
            }
            else
            {
                if (compute_lower_cell)
                {
                    int tmp = quadtree_cell_index_to_index_in_physical_arrays<SET_SIZE>(cell_index);
                    atomicAdd(&charge[tmp], particle_x * particle_charge);
                    atomicAdd(&charge[tmp], particle_y * particle_charge);
                    atomicAdd(&charge[tmp], particle_charge * particle_charge);

                    cell_index += quadtree_child_local_index_and_modify_cell_limits(
                        particle_x, particle_y,
                        current_cell_x_min, current_cell_x_max,
                        current_cell_y_min, current_cell_y_max);
                }

                int current_quadtree_value = quadtree[cell_index];

                if (current_quadtree_value == QUADTREE_LOCK)
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

                        float other_x = position_x[other_particle_index];
                        float other_y = position_y[other_particle_index];
                        float other_charge = charge[other_particle_index];

                        int other_next_child_number = quadtree_child_local_index(
                            other_x, other_y,
                            current_cell_x_min, current_cell_x_max,
                            current_cell_y_min, current_cell_y_max);

                        int other_new_parent_cell_numer = ((atomicAdd(allocated_quadcells_counter, 1) + 1) << 2);

                        quadtree[other_new_parent_cell_numer + other_next_child_number] = current_quadtree_value;
                        quadtree[cell_index] = other_new_parent_cell_numer;

                        int tmp = quadtree_cell_index_to_index_in_physical_arrays<SET_SIZE>(other_new_parent_cell_numer + other_next_child_number);
                        atomicAdd(&charge[tmp], other_x * other_charge);
                        atomicAdd(&charge[tmp], other_y * other_charge);
                        atomicAdd(&charge[tmp], other_charge);

                        // ? powinienem to robić ? chyba tak, bo czemu by nie
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
            }

            if (__syncthreads_and(success))
            {
                break;
            }
        }

        return;
    }

    template <int SET_SIZE>
    __global__ void compute_center_of_charge_kernel(float *position_x, float *position_y, float *charge)
    {
        int counter = *allocated_quadcells_counter;
        int index = threadIdx.x + blockIdx.x * blockDim.x;

        if (index < counter)
        {
            index += SET_SIZE;
            float c = charge[index];
            position_x[index] /= c;
            position_y[index] /= c;
        }
    }

    /*
    template <int SET_SIZE>
    __global__ void sort_kernel()
    {
        int index = threadIdx.x + blockIdx.x * blockDim.x;

        return;
    }
    */

    template <int SET_SIZE>
    __global__ void compute_velocities_kernel(
        float *position_x, float *position_y,
        float *velocity_x, float *velocity_y,
        float *charge, float *mass,
        int *quadtree)
    {
        int index = threadIdx.x + blockIdx.x * blockDim.x;

        float radius = 0.5 * (const_dev_x_max - const_dev_x_min);

        __shared__ float depth[STACK_SIZE * THREADS_PER_BLOCK / WARP_SIZE];
        __shared__ int stack[STACK_SIZE * THREADS_PER_BLOCK / WARP_SIZE];

        int counter = threadIdx.x % WARP_SIZE;
        int stack_start_index = STACK_SIZE * (threadIdx.x / WARP_SIZE);

        int jj = -1;
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            if (quadtree[i] != QUADTREE_EMPTY)
            {
                jj++;
            }
        }

        if (index < SET_SIZE)
        {
            float p_x = position_x[index];
            float p_y = position_y[index];

            float a_x = 0.0f;
            float a_y = 0.0f;

            int top = stack_start_index + jj;
            if (counter == 0)
            {
                int tmp = 0;

#pragma unroll
                for (int i = 0; i < 4; i++)
                {
                    if (quadtree[i] != QUADTREE_EMPTY)
                    {
                        stack[stack_start_index + tmp] = quadtree[i];
                        depth[stack_start_index + tmp] = radius * radius / THETA;
                        tmp++;
                    }
                    // if(child[i] == -1){
                    // 	printf("%s %d %d %d %d %s %d\n", "THROW ERROR!!!!", child[0], child[1], child[2], child[3], "top: ",top);
                    // }
                    // else{
                    // 	stack[stackStartIndex + temp] = child[i];
                    // 	depth[stackStartIndex + temp] = radius*radius/theta;
                    // 	temp++;
                    // }
                }
            }

            __syncthreads();

            // while stack is not empty
            while (top >= stack_start_index)
            {
                int node = stack[top];
                float dp = 0.25f * depth[top];
                // float dp = depth[top];

#pragma unroll
                for (int i = 0; i < 4; i++)
                {
                    int ch = QUADTREE_EMPTY;
                    if (node >= 0)
                    {
                        ch = quadtree[(node << 2) + i];
                    }

                    if (ch != QUADTREE_EMPTY)
                    {
                        if (ch < 0)
                        {
                            ch = ~ch;
                        }
                        else
                        {
                            ch += SET_SIZE;
                        }

                        float dx = position_x[ch] - p_x;
                        float dy = position_y[ch] - p_y;
                        float r = dx * dx + dy * dy + EPS;
                        if (ch < SET_SIZE /*is leaf node*/ || __all_sync((unsigned int)0xFFFFFFFF, dp <= r) /*meets criterion*/)
                        {
                            r = rsqrt(r);

                            // + if they repel, - if they attract
                            float f = K * charge[ch] * charge[index] * r * r * r / (mass[index]);



                            a_x += -f * dx;
                            a_y += -f * dy;
                        }
                        else
                        {
                            if (counter == 0)
                            {
                                stack[top] = ch;
                                //depth[top] = dp;
                                depth[top] = 0.25*dp;
                            }
                            top++;
                            //__threadfence();
                        }
                    }
                }

                top--;
            }

            velocity_x[index] += SCALE_MULTIPLIER * a_x;
            velocity_y[index] += SCALE_MULTIPLIER * a_y;
        }
    }

    __device__ inline void set_color(float f, GLubyte pixel[4])
    {
        pixel[0] = 0x00;
        pixel[1] = 0x00;
        pixel[2] = 0x00;
        pixel[3] = 0x00;

        f *= 0.05f;

        if (f > 0)
        {
            pixel[0] = (unsigned char) ((f > 255 ? 255 : f));
        }
        else if (f < 0)
        {
            pixel[1] = (unsigned char) ((f < -255 ? 255 : -f));
        }
    }

    template <int SET_SIZE>
    __global__ void compute_pixels_kernel(
        float* position_x, float* position_y,
        float* charge, int* quadtree,
        GLubyte* pixel_buffer,
        float x_min, float x_max, float y_min, float y_max,
        int width, int height)
    {
        int index = threadIdx.x + blockIdx.x * blockDim.x;

        if (index < width * height)
        {
            float radius = 0.5 * (const_dev_x_max - const_dev_x_min);

            __shared__ float depth[STACK_SIZE * THREADS_PER_BLOCK / WARP_SIZE];
            __shared__ int stack[STACK_SIZE * THREADS_PER_BLOCK / WARP_SIZE];

            int counter = threadIdx.x % WARP_SIZE;
            int stack_start_index = STACK_SIZE * (threadIdx.x / WARP_SIZE);

            int jj = -1;
#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                if (quadtree[i] != QUADTREE_EMPTY)
                {
                    jj++;
                }
            }

            if (index < height * width)
            {
                int x = index % width;
                int y = index / width;

                float p_x = (float)x / (float)width;
                float p_y = (float)y / (float)height;

                float f_x = 0.0f;
                float f_y = 0.0f;

                int top = stack_start_index + jj;
                if (counter == 0)
                {
                    int tmp = 0;

#pragma unroll
                    for (int i = 0; i < 4; i++)
                    {
                        if (quadtree[i] != QUADTREE_EMPTY)
                        {
                            stack[stack_start_index + tmp] = quadtree[i];
                            depth[stack_start_index + tmp] = radius * radius / THETA;
                            tmp++;
                        }
                        // if(child[i] == -1){
                        // 	printf("%s %d %d %d %d %s %d\n", "THROW ERROR!!!!", child[0], child[1], child[2], child[3], "top: ",top);
                        // }
                        // else{
                        // 	stack[stackStartIndex + temp] = child[i];
                        // 	depth[stackStartIndex + temp] = radius*radius/theta;
                        // 	temp++;
                        // }
                    }
                }

                __syncthreads();

                // while stack is not empty
                while (top >= stack_start_index)
                {
                    int node = stack[top];
                    float dp = 0.25f * depth[top];
                    // float dp = depth[top];

#pragma unroll
                    for (int i = 0; i < 4; i++)
                    {
                        int ch = QUADTREE_EMPTY;
                        if (node >= 0)
                        {
                            ch = quadtree[(node << 2) + i];
                        }

                        if (ch != QUADTREE_EMPTY)
                        {
                            if (ch < 0)
                            {
                                ch = ~ch;
                            }
                            else
                            {
                                ch += SET_SIZE;
                            }

                            float dx = position_x[ch] - p_x;
                            float dy = position_y[ch] - p_y;
                            float r = dx * dx + dy * dy + EPS;
                            if (ch < SET_SIZE /*is leaf node*/ || __all_sync((unsigned int)0xFFFFFFFF, dp <= r) /*meets criterion*/)
                            {
                                r = rsqrt(r);

                                // + if they repel, - if they attract
                                float f = K * charge[ch] * r * r * r;



                                f_x += -f * dx;
                                f_y += -f * dy;
                            }
                            else
                            {
                                if (counter == 0)
                                {
                                    stack[top] = ch;
                                    //depth[top] = dp;
                                    depth[top] = 0.25 * dp;
                                }
                                top++;
                                //__threadfence();
                            }
                        }
                    }

                    top--;
                }


                GLubyte pixel[4];
                pixel_buffer[index * 4] = 255;
                pixel_buffer[index * 4 + 1] = 0;
                pixel_buffer[index * 4 + 2] = 0;
                pixel_buffer[index * 4 + 3] = 0;
                
                
                set_color(sqrt(f_x * f_x + f_y * f_y), pixel);
                ((GLuint*)(pixel_buffer))[index] = *((GLuint *)pixel);

                //buffer[index] = *((GLuint *)pixel);
                
                /*
                unsigned int green = 255;
                unsigned int red = 0;
                unsigned int blue = 255;

                unsigned int aaa = (green << 12) + (red << 4) + (blue << 8);

                buffer[index] = *((GLuint *)&aaa);
                */

            }
        }
    }
}