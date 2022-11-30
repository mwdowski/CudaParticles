#include "application.hpp"

const char *application::WINDOW_TITLE = "GPU PROJECT";
const int application::WINDOW_POSITION_X = -1;
const int application::WINDOW_POSITION_Y = -1;
const int application::WINDOW_SIZE_X = 1366;
const int application::WINDOW_SIZE_Y = 768;
GLubyte *application::PixelBuffer = new GLubyte[WINDOW_SIZE_X * WINDOW_SIZE_Y * sizeof(uint)];

void application::start(int &argc, char *argv[])
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowPosition(WINDOW_POSITION_X, WINDOW_POSITION_Y);
    glutInitWindowSize(WINDOW_SIZE_X, WINDOW_SIZE_Y);
    glutCreateWindow(WINDOW_TITLE);

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    makePixel(21, 37, 0, 255, 255, PixelBuffer);
    glutDisplayFunc(display);
    glutMainLoop();
}

void application::display()
{
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawPixels(WINDOW_SIZE_X, WINDOW_SIZE_Y, GL_RGBA, GL_UNSIGNED_BYTE, PixelBuffer);
    glutSwapBuffers();
    glFlush();
}

void application::makePixel(int x, int y, int r, int g, int b, GLubyte *pixels)
{
    int position = (y * WINDOW_SIZE_X + x) * sizeof(uint);
    application::PixelBuffer[position] = r;
    application::PixelBuffer[position + 1] = g;
    application::PixelBuffer[position + 2] = b;
}
