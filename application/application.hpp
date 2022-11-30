#pragma once

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

class application
{
    static const char *WINDOW_TITLE;
    static const int WINDOW_SIZE_X;
    static const int WINDOW_SIZE_Y;
    static const int WINDOW_POSITION_X;
    static const int WINDOW_POSITION_Y;
    static GLubyte *PixelBuffer;

public:
    static void start(int &argc, char *argv[]);
    application() = delete;
    ~application() = delete;

private:
    static void display();
    static void makePixel(int x, int y, int r, int g, int b, GLubyte *pixels);
};