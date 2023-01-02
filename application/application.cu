#include "application.cuh"
#include <stdio.h>
#include "../macros/macros.cuh"

const char *application::WINDOW_TITLE = "GPU PROJECT";
const int application::WINDOW_POSITION_X = -1;
const int application::WINDOW_POSITION_Y = -1;
const int application::WINDOW_SIZE_X = 1366;
const int application::WINDOW_SIZE_Y = 768;
const float application::WINDOW_SIZE_ASPECT = (float)WINDOW_SIZE_X / (float)WINDOW_SIZE_Y;
GLubyte *application::PixelBuffer = new GLubyte[WINDOW_SIZE_X * WINDOW_SIZE_Y * 4];
std::unique_ptr<particles::particles_set<application::PARTICLES_NUMBER>> application::arr =
    std::unique_ptr<particles::particles_set<application::PARTICLES_NUMBER>>(particles::particles_set<application::PARTICLES_NUMBER>::generate());
particles::engine<application::PARTICLES_NUMBER> &application::eng = particles::engine<application::PARTICLES_NUMBER>::instance();
float application::x_min = 0;
float application::x_max = 0;
float application::y_min = 0;
float application::y_max = 0;
int application::width = 0;
int application::height = 0;
int application::milliseconds_between_refresh = 0;

bool pause = false;

void onexit()
{
    printf("\n");
}

void mouse_right(int x, int y)
{
    application::eng.set_mouse_particle(
        ((float)x / (float)application::width) * (application::x_max - application::x_min) + application::x_min,
        -((float)y / (float)application::height) * (application::y_max - application::y_min) - application::y_min,
        50.0f);
}

void mouse_left(int x, int y)
{
    application::eng.set_mouse_particle(
        ((float)x / (float)application::width) * (application::x_max - application::x_min) + application::x_min,
        -((float)y / (float)application::height) * (application::y_max - application::y_min) - application::y_min,
        -50.0f);
}

void mouse_middle(int x, int y)
{
    application::eng.set_mouse_particle(
        ((float)x / (float)application::width) * (application::x_max - application::x_min) + application::x_min,
        -((float)y / (float)application::height) * (application::y_max - application::y_min) - application::y_min,
        0.0f);
}

void mouse_move(int button, int state, int x, int y)
{
    if (state == GLUT_UP)
    {
        if (button == GLUT_RIGHT_BUTTON)
        {
            mouse_right(x, y);
        }
        else if (button == GLUT_LEFT_BUTTON)
        {
            mouse_left(x, y);
        }
        else if (button == GLUT_MIDDLE_BUTTON)
        {
            mouse_middle(x, y);
        }
    }
}

void application::keyboard(unsigned char c, int x, int y)
{
    if (c == ' ')
    {
        pause = !pause;
        if (!pause)
        {
            glutTimerFunc(0, application::timer, 0);
        }
    }
}

void application::start(int &argc, char *argv[])
{
    eng.initiate();

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowPosition(WINDOW_POSITION_X, WINDOW_POSITION_Y);
    glutInitWindowSize(WINDOW_SIZE_X, WINDOW_SIZE_Y);
    atexit(onexit);

    glutCreateWindow(WINDOW_TITLE);

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    glutMouseFunc(mouse_move);
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard);

    cuda_try_or_exit(eng.load_data_to_gpu(arr.get()));
    glutTimerFunc(milliseconds_between_refresh, timer, 0);

    glutMainLoop();
}

void application::display()
{
    glFinish();
    //printf("1. started drawing\n");
    glClear(GL_COLOR_BUFFER_BIT);

    glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, PixelBuffer);

    glBegin(GL_POINTS);
    for (int i = 0; i < arr->size; i++)
    {
        if (arr->charge[i] > 0)
        {
            glColor3f(1, 1, 1);
        }
        else
        {
            glColor3f(0, 1, 0);
        }
        glVertex2f(arr->position_x[i], arr->position_y[i]);
    }
    glEnd();

    glutSwapBuffers();
    //printf("2. ended drawing\n");
    glFinish();
    // glFlush();
}

void application::reshape(int width, int height)
{
    glMatrixMode(GL_PROJECTION);

    glViewport(0, 0, width, height);

    float width_size = (float)width / WINDOW_SIZE_X;
    float height_size = (float)height / WINDOW_SIZE_Y;

    application::width = width;
    application::height = height;

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
    glFinish();
    glutPostRedisplay();

    //printf("3. started counting\n");
    int new_time = glutGet(GLUT_ELAPSED_TIME);
    printf("\r%f   ", 1000.0f / (new_time - time2137));
    fflush(stdout);
    time2137 = new_time;

    cuda_try_or_exit(eng.move(x_min, x_max, y_min, y_max, width, height));
    cuda_try_or_exit(eng.load_data_from_gpu(arr.get(), PixelBuffer));

    //printf("4. ended counting\n");

    if (!pause)
    {
        glutTimerFunc(milliseconds_between_refresh, timer, 0);
    }

    glFinish();
}

void application::makePixel(int x, int y, int r, int g, int b, GLubyte *pixels)
{
    int position = (y * WINDOW_SIZE_X + x) * sizeof(unsigned int);
    application::PixelBuffer[position] = r;
    application::PixelBuffer[position + 1] = g;
    application::PixelBuffer[position + 2] = b;
}
