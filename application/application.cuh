#pragma once

//#define GLFW_DLL
#include <GLFW/glfw3.h>

#include "../particles/particles_set/particles_set.hpp"
#include "../particles/engine/engine.cuh"

class application
{
public:
    static const int PARTICLES_NUMBER = 10'000;
    static const char *WINDOW_TITLE;
    static const int WINDOW_SIZE_X;
    static const int WINDOW_SIZE_Y;
    static const float WINDOW_SIZE_ASPECT;
    static const int WINDOW_POSITION_X;
    static const int WINDOW_POSITION_Y;
    static float x_min;
    static float x_max;
    static float y_min;
    static float y_max;
    static std::unique_ptr<particles::particles_set<PARTICLES_NUMBER>> arr;
    static unsigned char *PixelBuffer;
    static particles::engine<PARTICLES_NUMBER> &eng;
    static int milliseconds_between_refresh;

    static int width;
    static int height;

public:
    static void start(int &argc, char *argv[]);
    application() = delete;
    ~application() = delete;

private:
    static void display(GLFWwindow *window);
    static void reshape(int width, int height);
    static void timer();
    static void makePixel(int x, int y, int r, int g, int b, GLubyte *pixels);
    static void keyboard(unsigned char c, int x, int y);
};