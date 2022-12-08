#pragma once

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include "../particles/particles_set/particles_set.hpp"
#include "../particles/engine/engine.cuh"

class application
{
public:
    static const int PARTICLES_NUMBER = 10'000;
    static const char *WINDOW_TITLE;
    static const int WINDOW_SIZE_X;
    static const int WINDOW_SIZE_Y;
    static const double WINDOW_SIZE_ASPECT;
    static const int WINDOW_POSITION_X;
    static const int WINDOW_POSITION_Y;
    static std::unique_ptr<particles::particles_set<PARTICLES_NUMBER>> arr;
    static GLubyte *PixelBuffer;
    static particles::engine<PARTICLES_NUMBER> &eng;

public:
    static void start(int &argc, char *argv[]);
    application() = delete;
    ~application() = delete;

private:
    static void display();
    static void reshape(int width, int height);
    static void timer(int value);
    static void makePixel(int x, int y, int r, int g, int b, GLubyte *pixels);
};