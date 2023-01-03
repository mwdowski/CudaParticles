#include "application.cuh"
#include <stdio.h>
#include "../macros/macros.cuh"
#include <time.h>
#include <cuda_gl_interop.h>

const char *application::WINDOW_TITLE = "GPU PROJECT";
const int application::WINDOW_POSITION_X = -1;
const int application::WINDOW_POSITION_Y = -1;
const int application::WINDOW_SIZE_X = 1366;
const int application::WINDOW_SIZE_Y = 768;
const float application::WINDOW_SIZE_ASPECT = (float)WINDOW_SIZE_X / (float)WINDOW_SIZE_Y;
unsigned char *application::PixelBuffer = new unsigned char[WINDOW_SIZE_X * WINDOW_SIZE_Y * 4];
std::unique_ptr<particles::particles_set<application::PARTICLES_NUMBER>> application::arr =
    std::unique_ptr<particles::particles_set<application::PARTICLES_NUMBER>>(particles::particles_set<application::PARTICLES_NUMBER>::generate());
particles::engine<application::PARTICLES_NUMBER> &application::eng = particles::engine<application::PARTICLES_NUMBER>::instance();
float application::x_min = 0;
float application::x_max = 0;
float application::y_min = 0;
float application::y_max = 0;
int application::width = 0;
int application::height = 0;
int application::milliseconds_between_refresh = 30;

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
        200.0f);
}

void mouse_left(int x, int y)
{
    application::eng.set_mouse_particle(
        ((float)x / (float)application::width) * (application::x_max - application::x_min) + application::x_min,
        -((float)y / (float)application::height) * (application::y_max - application::y_min) - application::y_min,
        -200.0f);
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
}

void application::keyboard(unsigned char c, int x, int y)
{
    if (c == ' ')
    {
        pause = !pause;
        if (!pause)
        {
        }
    }
}

GLuint texture_id;
cudaGraphicsResource *m_cudaGraphicsResource;
cudaTextureObject_t m_texture;
GLuint buffer_id;

void register_buffer()
{
    glGenTextures(1, &texture_id);
    glBindTexture(GL_TEXTURE_2D, texture_id);

    /*
    // set basic parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    */

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1366, 768, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    cuda_try_or_exit(cudaGraphicsGLRegisterImage(&m_cudaGraphicsResource, texture_id, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard));
}

void error_callback(int error, const char *description)
{
    fprintf(stderr, "Error: %s\n", description);
}

void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
    {
        pause = !pause;
    }
}

void mouse_button_callback(GLFWwindow *window, int button, int action, int mods)
{
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
    {
        mouse_left(xpos, ypos);
    }
    else if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS)
    {
        mouse_right(xpos, ypos);
    }
    else if (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS)
    {
        mouse_middle(xpos, ypos);
    }
}

void application::start(int &argc, char *argv[])
{
    eng.initiate();
    eng.load_data_to_gpu(arr.get());
    if (!glfwInit())
    {
        return;
    }

    glfwSetErrorCallback(error_callback);

    GLFWwindow *window = glfwCreateWindow(WINDOW_SIZE_X, WINDOW_SIZE_Y, WINDOW_TITLE, NULL, NULL);
    if (!window)
    {
        return;
    }

    glfwMakeContextCurrent(window);

    glEnable(GL_TEXTURE_2D);
    glfwSetKeyCallback(window, key_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    while (!glfwWindowShouldClose(window))
    {
        reshape(1366, 768);
        display(window);
        if (!pause)
        {
            timer();
        }

        glfwPollEvents();
    }

    onexit();
    glfwDestroyWindow(window);
    glfwTerminate();
}

void application::display(GLFWwindow *window)
{
    glFinish();
    // printf("1. started drawing\n");
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, PixelBuffer);

    /*
    GLuint texID = 2137;
    if (texID == 2137)
        glGenTextures(1, &texID);
    glBindTexture(GL_TEXTURE_2D, texID);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1366, 768, 0, GL_RGBA, GL_UNSIGNED_BYTE, PixelBuffer);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    glBegin(GL_QUADS);
    glTexCoord2f(0, 0);
    glVertex2f(x_min, y_min);
    glTexCoord2f(1, 0);
    glVertex2f(x_max, y_min);
    glTexCoord2f(1, 1);
    glVertex2f(x_max, y_max);
    glTexCoord2f(0, 1);
    glVertex2f(x_min, y_max);
    glEnd();
    */

    // glBindTexture(GL_TEXTURE_2D, 0);

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

    glfwSwapBuffers(window);
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

double time2137 = 0;

void application::timer()
{

    double new_time = glfwGetTime();

    printf("\r%f   ", 1.0 / (new_time - time2137));
    fflush(stdout);
    time2137 = new_time;

    cuda_try_or_exit(eng.move(x_min, x_max, y_min, y_max, width, height, m_cudaGraphicsResource));
    cuda_try_or_exit(eng.load_data_from_gpu(arr.get(), PixelBuffer));
}

void application::makePixel(int x, int y, int r, int g, int b, GLubyte *pixels)
{
    int position = (y * WINDOW_SIZE_X + x) * sizeof(unsigned int);
    application::PixelBuffer[position] = r;
    application::PixelBuffer[position + 1] = g;
    application::PixelBuffer[position + 2] = b;
}
