#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <mujoco/mjmodel.h>
#include <mujoco/mjrender.h>
#include <mujoco/mjspec.h>
#include <mujoco/mjtnum.h>
#include <mujoco/mjvisualize.h>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <thread>

#include "opencv2/opencv.hpp"
#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

// MuJoCo data structures
mjModel *m = NULL; // MuJoCo model
mjData *d = NULL;  // MuJoCo data
mjvCamera cam;     // abstract camera
mjvOption opt;     // visualization options
mjvScene scn;      // abstract scene
mjrContext con;    // custom GPU context

// mouse interaction
bool button_left = false;
bool button_middle = false;
bool button_right = false;
double lastx = 0;
double lasty = 0;

// keyboard callback
void keyboard(GLFWwindow *window, int key, int scancode, int act, int mods) {
  // backspace: reset simulation
  if (act == GLFW_PRESS && key == GLFW_KEY_BACKSPACE) {
    mj_resetData(m, d);
    mj_forward(m, d);
  }
}

// mouse button callback
void mouse_button(GLFWwindow *window, int button, int act, int mods) {
  // update button state
  button_left =
      (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
  button_middle =
      (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS);
  button_right =
      (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);

  // update mouse position
  glfwGetCursorPos(window, &lastx, &lasty);
}

// mouse move callback
void mouse_move(GLFWwindow *window, double xpos, double ypos) {
  // no buttons down: nothing to do
  if (!button_left && !button_middle && !button_right) {
    return;
  }

  // compute mouse displacement, save
  double dx = xpos - lastx;
  double dy = ypos - lasty;
  lastx = xpos;
  lasty = ypos;

  // get current window size
  int width, height;
  glfwGetWindowSize(window, &width, &height);

  // get shift key state
  bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                    glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);

  // determine action based on mouse button
  mjtMouse action;
  if (button_right) {
    action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
  } else if (button_left) {
    action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
  } else {
    action = mjMOUSE_ZOOM;
  }

  // move camera
  mjv_moveCamera(m, action, dx / height, dy / height, &scn, &cam);
}

// scroll callback
void scroll(GLFWwindow *window, double xoffset, double yoffset) {
  // emulate vertical mouse motion = 5% of window height
  mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05 * yoffset, &scn, &cam);
}

void get_sensor_data(const mjModel *model, const mjData *data,
                     const std::string &sensor_name, mjtNum *sensor_data) {
  int sensor_id = mj_name2id(model, mjOBJ_SENSOR, sensor_name.c_str());
  if (sensor_id == -1) {
    std::cout << "no found sensor" << std::endl;
    return;
  }
  int data_pos = model->sensor_adr[sensor_id];

  for (int i = 0; i < model->sensor_dim[sensor_id]; i++) {
    sensor_data[i] = data->sensordata[data_pos + i];
  }
}

void get_cam_image(mjvCamera *cam, int width, int height, int stereo) {
  mjrRect viewport2 = {0, 0, width, height};
  int before_stereo = scn.stereo;
  scn.stereo = stereo;
  // mujoco更新渲染
  mjv_updateCamera(m, d, cam, &scn);
  mjr_render(viewport2, &scn, &con);
  scn.stereo = before_stereo;
  // 渲染完成读取图像
  unsigned char *rgbBuffer = new unsigned char[width * height * 3];
  float *depthBuffer = new float[width * height];
  mjr_readPixels(rgbBuffer, depthBuffer, viewport2, &con);
  cv::Mat image(height, width, CV_8UC3, rgbBuffer);
  // 反转图像以匹配OpenGL渲染坐标系
  cv::flip(image, image, 0);
  // 颜色顺序转换这样要使用bgr2rgb而不是rgb2bgr
  cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
  cv::imshow("Image", image);
  cv::waitKey(1);
  // 释放内存
  delete[] rgbBuffer;
  delete[] depthBuffer;
}

// main function
int main(int argc, const char **argv) {

  char error[1000] = "Could not load binary model";
  m = mj_loadXML("../touch_pad.xml", 0, error, 1000);

  // make data
  d = mj_makeData(m);

  // init GLFW
  if (!glfwInit()) {
    mju_error("Could not initialize GLFW");
  }

  // create window, make OpenGL context current, request v-sync
  GLFWwindow *window = glfwCreateWindow(1200, 900, "Demo", NULL, NULL);
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);

  // initialize visualization data structures
  mjv_defaultCamera(&cam);
  mjv_defaultOption(&opt);
  mjv_defaultScene(&scn);
  mjr_defaultContext(&con);

  // create scene and context
  mjv_makeScene(m, &scn, 20000);
  mjr_makeContext(m, &con, mjFONTSCALE_150);

  // install GLFW mouse and keyboard callbacks
  glfwSetKeyCallback(window, keyboard);
  glfwSetCursorPosCallback(window, mouse_move);
  glfwSetMouseButtonCallback(window, mouse_button);
  glfwSetScrollCallback(window, scroll);

  /*--------可视化配置--------*/
  opt.flags[mjtVisFlag::mjVIS_CONTACTPOINT] = true;
  opt.flags[mjtVisFlag::mjVIS_CONTACTFORCE] = true;
  // opt.flags[mjtVisFlag::mjVIS_CAMERA] = true;
  // opt.flags[mjtVisFlag::mjVIS_CONVEXHULL] = true;
  // opt.flags[mjtVisFlag::mjVIS_COM] = true;
  // opt.label = mjtLabel::mjLABEL_BODY;
  opt.frame = mjtFrame::mjFRAME_WORLD;
  /*--------可视化配置--------*/

  /*--------场景渲染--------*/
  scn.flags[mjtRndFlag::mjRND_WIREFRAME] = true;
  // scn.flags[mjtRndFlag::mjRND_SEGMENT] = true;
  // scn.flags[mjtRndFlag::mjRND_IDCOLOR] = true;
  /*--------场景渲染--------*/

  /*--------读取传感器--------*/
  std::cout << "sensor_num: " << m->nsensor << std::endl;
  auto print = [&](int num, int *adr) {
    std::cout << "num:" << num << std::endl;
    std::cout << "name:";
    for (int i = 0; i < num; i++)
      std::cout << "  " << m->names + adr[i];
    std::cout << std::endl;
  };
  // print(m->nsensor, m->name_sensoradr);
  // 获取touch_point0000位置
  int sensor_id = mj_name2id(m, mjOBJ_SENSOR, "touch_point0000");
  if (sensor_id == -1) {
    std::cout << "no found sensor" << std::endl;
  }
  
  int touch_point_adr[20][20];
  for (int x = 0; x < 20; x++) {
    for (int y = 0; y < 20; y++) {
      int idx = x * 20 + y ;
      touch_point_adr[x][y] = m->sensor_adr[idx];
    }
  }
  std::cout << std::endl;
    while (!glfwWindowShouldClose(window)) {
      auto step_start = std::chrono::high_resolution_clock::now();

      mj_step(m, d);

      int cnt = 0;
      cv::Mat image(20, 20, CV_8UC1);
      for (int x = 0; x < 20; x++) {
        for (int y = 0; y < 20; y++) {
          int idx = x * 20 + y ;
          mjtNum data = mju_norm3(d->sensordata + touch_point_adr[x][y]);
          data = mju_clip(data, 0.0, 3.0);
          if (data > 0.5) {
            cnt++;
          }
          image.at<uchar>(x, y) = data / 3 * 255;
        }
      }
      // std::cout << cnt << std::endl;
      cv::resize(image, image, cv::Size(200, 200));
      cv::imshow("touch", image);
      cv::waitKey(1);

      // auto force = get_sensor_data(m, d, "touch_point1900");
      // std::cout << "force: " << force[0] << " " << force[1] << " " <<
      // force[2]
      //           << std::endl;

      // get framebuffer viewport
      mjrRect viewport = {0, 0, 0, 0};
      glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

      // update scene and render
      mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
      mjr_render(viewport, &scn, &con);

      // swap OpenGL buffers (blocking call due to v-sync)
      glfwSwapBuffers(window);

      // process pending GUI events, call GLFW callbacks
      glfwPollEvents();

      //同步时间
      auto current_time = std::chrono::high_resolution_clock::now();
      double elapsed_sec =
          std::chrono::duration<double>(current_time - step_start).count();
      double time_until_next_step = m->opt.timestep - elapsed_sec;
      if (time_until_next_step > 0.0) {
        auto sleep_duration =
            std::chrono::duration<double>(time_until_next_step);
        std::this_thread::sleep_for(sleep_duration);
      }
    }

    // free visualization storage
    mjv_freeScene(&scn);
    mjr_freeContext(&con);

    // free MuJoCo model and data
    mj_deleteData(d);
    mj_deleteModel(m);

    // terminate GLFW (crashes with Linux NVidia drivers)
#if defined(__APPLE__) || defined(_WIN32)
    glfwTerminate();
#endif

    return 1;
  }
