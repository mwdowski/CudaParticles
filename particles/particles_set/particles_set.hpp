#pragma once

#include <stdio.h>

namespace particles
{
    template <int SET_SIZE>
    class particles_set
    {

    public:
        const int size = SET_SIZE;
        float *position_x;
        float *position_y;
        float *velocity_x;
        float *velocity_y;
        float *charge;
        float *mass;

    public:
        static particles_set<SET_SIZE> *generate()
        {
            srand(2137);
            particles_set<SET_SIZE> *result = new particles_set<SET_SIZE>();
            for (int i = 0; i < SET_SIZE; i++)
            {
                result->position_x[i] = random_float(-1.5, -0.5);
                result->position_y[i] = random_float(-1.5, -0.5);
                result->velocity_x[i] = 0.0f;//random_float(-0.001, 0.001);
                result->velocity_y[i] = 0;// random_float(-0.001, 0.001);

                //if (*random_one_or_other('p', 'e') == 'p')
                //{
                    result->charge[i] = 1;
                    result->mass[i] = 1.00727647;
                /*
                }
                else
                {
                    result->charge[i] = -1;
                    result->mass[i] = 0.0005485;
                }
                */
            }

            return result;
        }

        void move()
        {
            for (int i = 0; i < SET_SIZE; i++)
            {
                this->position_x[i] += this->velocity_x[i];
                this->position_y[i] += this->velocity_y[i];
            }
        }

        ~particles_set()
        {
            delete[] this->position_x;
            delete[] this->position_y;
            delete[] this->velocity_x;
            delete[] this->velocity_y;
            delete[] this->charge;
            delete[] this->mass;
        }

    private:
        static float random_float(float min, float max)
        {
            return min + (float)rand() / ((float)RAND_MAX / (max - min));
        }

        template <class T>
        static T random_one_or_other(T one, T other)
        {
            return rand() % 2 == 0 ? one : other;
        }

        particles_set()
        {
            this->position_x = new float[SET_SIZE];
            this->position_y = new float[SET_SIZE];
            this->velocity_x = new float[SET_SIZE];
            this->velocity_y = new float[SET_SIZE];
            this->charge = new float[SET_SIZE];
            this->mass = new float[SET_SIZE];
        }
    };
}