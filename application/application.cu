#include "application.cuh"
#include <stdio.h>
#include "../macros/macros.cuh"

const char *application::WINDOW_TITLE = "GPU PROJECT";
const int application::WINDOW_POSITION_X = -1;
const int application::WINDOW_POSITION_Y = -1;
const int application::WINDOW_SIZE_X = 1366;
const int application::WINDOW_SIZE_Y = 768;
const double application::WINDOW_SIZE_ASPECT = (double)WINDOW_SIZE_X / (double)WINDOW_SIZE_Y;
GLubyte *application::PixelBuffer = new GLubyte[WINDOW_SIZE_X * WINDOW_SIZE_Y * sizeof(uint)];
std::unique_ptr<particles::particles_set<application::PARTICLES_NUMBER>> application::arr =
    std::unique_ptr<particles::particles_set<application::PARTICLES_NUMBER>>(particles::particles_set<application::PARTICLES_NUMBER>::generate());
particles::engine<application::PARTICLES_NUMBER> &application::eng = particles::engine<application::PARTICLES_NUMBER>::instance();
float application::x_min = 0;
float application::x_max = 0;
float application::y_min = 0;
float application::y_max = 0;

void application::start(int &argc, char *argv[])
{
    eng.initiate();

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowPosition(WINDOW_POSITION_X, WINDOW_POSITION_Y);
    glutInitWindowSize(WINDOW_SIZE_X, WINDOW_SIZE_Y);

    glutCreateWindow(WINDOW_TITLE);

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);

    cuda_try_or_exit(eng.load_data_to_gpu(arr.get()));
    glutTimerFunc(0, timer, 0);

    glutMainLoop();
}

void application::display()
{
    glClear(GL_COLOR_BUFFER_BIT);

    glBegin(GL_POINTS);
    for (int i = 0; i < arr->size; i++)
    {
        glVertex2f(arr->position_x[i], arr->position_y[i]);
    }
    glEnd();
    // glDrawPixels(WINDOW_SIZE_X, WINDOW_SIZE_Y, GL_RGBA, GL_UNSIGNED_BYTE, PixelBuffer);
    glutSwapBuffers();
    // glFlush();
}

void application::reshape(int width, int height)
{
    glMatrixMode(GL_PROJECTION);

    glViewport(0, 0, width, height);

    double width_size = (double)width / WINDOW_SIZE_X;
    double height_size = (double)height / WINDOW_SIZE_Y;

    glLoadIdentity();
    x_min = -width_size * WINDOW_SIZE_ASPECT;
    x_max = width_size * WINDOW_SIZE_ASPECT;
    y_min = -height_size;
    y_max = height_size;
    glOrtho(x_min, x_max, y_min, y_max, -1.0, 1.0);

    glMatrixMode(GL_MODELVIEW);
}

int time2137 = 0;

void application::timer(int value)
{
    int new_time = glutGet(GLUT_ELAPSED_TIME);
    printf("%f\n", 1000.0f / (new_time - time2137));
    time2137 = new_time;

    GLfloat model[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, model);

    cuda_try_or_exit(eng.move(x_min, x_max, y_min, y_max));
    cuda_try_or_exit(eng.load_data_from_gpu(arr.get()));
    glutPostRedisplay();

    glutTimerFunc(0, timer, 0);
}

void application::makePixel(int x, int y, int r, int g, int b, GLubyte *pixels)
{
    int position = (y * WINDOW_SIZE_X + x) * sizeof(uint);
    application::PixelBuffer[position] = r;
    application::PixelBuffer[position + 1] = g;
    application::PixelBuffer[position + 2] = b;
}
