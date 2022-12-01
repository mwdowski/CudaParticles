#include <stdio.h>
#include "../application/application.hpp"
//#include "../particles/particles_set/particles_set.hpp"

int main(int argc, char *argv[])
{
    /*
    particles::particles_set<10> ps = particles::particles_set<10>::generate();
    for (int i = 0; i < ps.size; i++)
    {
        printf("%f, %f\n", ps.mass[i], ps.charge[i]);
    }
    */
    application::start(argc, argv);

    return 0;
}