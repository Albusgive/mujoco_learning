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
#include <thread>

#include "opencv2/opencv.hpp"
#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>

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

std::vector<float> get_sensor_data(const mjModel *model, const mjData *data,
                                   const std::string &sensor_name) {
  int sensor_id = mj_name2id(model, mjOBJ_SENSOR, sensor_name.c_str());
  if (sensor_id == -1) {
    std::cout << "no found sensor" << std::endl;
    return std::vector<float>();
  }
  int data_pos = model->sensor_adr[sensor_id];
  std::vector<float> sensor_data(model->sensor_dim[sensor_id]);
  for (int i = 0; i < sensor_data.size(); i++) {
    sensor_data[i] = data->sensordata[data_pos + i];
  }
  return sensor_data;
}

void get_cam_image(mjvCamera *cam,int width,int height,int stereo) {
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
  m = mj_loadXML("../../../API-MJC/vis_cfg.xml", 0, error, 1000);

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
  mjv_makeScene(m, &scn, 2000);
  mjr_makeContext(m, &con, mjFONTSCALE_150);

  // install GLFW mouse and keyboard callbacks
  glfwSetKeyCallback(window, keyboard);
  glfwSetCursorPosCallback(window, mouse_move);
  glfwSetMouseButtonCallback(window, mouse_button);
  glfwSetScrollCallback(window, scroll);

  /*--------可视化配置--------*/
  // opt.flags[mjtVisFlag::mjVIS_CONTACTPOINT] = true;
  // opt.flags[mjtVisFlag::mjVIS_CAMERA] = true;
  // opt.flags[mjtVisFlag::mjVIS_CONVEXHULL] = true;
  // opt.flags[mjtVisFlag::mjVIS_COM] = true;
  // opt.label = mjtLabel::mjLABEL_BODY;
  // opt.frame = mjtFrame::mjFRAME_BODY;
  /*--------可视化配置--------*/


  /*--------场景渲染--------*/
  // scn.flags[mjtRndFlag::mjRND_WIREFRAME] = true;
  // scn.flags[mjtRndFlag::mjRND_SEGMENT] = true;
  // scn.flags[mjtRndFlag::mjRND_IDCOLOR] = true;
  int bunny_id = mj_name2id(m, mjOBJ_GEOM, "bunny");
  /*--------场景渲染--------*/

  /*--------单/双目渲染--------*/
  scn.stereo = mjtStereo::mjSTEREO_SIDEBYSIDE;
  /*--------单/双目渲染--------*/

  //相机初始化
  mjvCamera cam2;
  int camID = mj_name2id(m, mjOBJ_CAMERA, "bunny_eyes");
  if (camID == -1) {
    std::cerr << "Camera not found" << std::endl;
  } else {
    mjv_defaultCamera(&cam2);
    cam2.fixedcamid = camID;
    cam2.type = mjCAMERA_FIXED;
  }

  auto step_start = std::chrono::high_resolution_clock::now();
  while (!glfwWindowShouldClose(window)) {

    mj_step(m, d);

    get_cam_image(&cam2,1024,640,mjtStereo::mjSTEREO_SIDEBYSIDE);

    //同步时间
    auto current_time = std::chrono::high_resolution_clock::now();
    double elapsed_sec =
        std::chrono::duration<double>(current_time - step_start).count();
    double time_until_next_step = m->opt.timestep * 5 - elapsed_sec;
    if (time_until_next_step > 0.0) {
      auto sleep_duration = std::chrono::duration<double>(time_until_next_step);
      std::this_thread::sleep_for(sleep_duration);
    }

    // get framebuffer viewport
    mjrRect viewport = {0, 0, 0, 0};
    glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

    // update scene and render
    mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
    /*--------设置分割颜色--------*/
    mjvGeom *geom;
    // std::cout << scn.ngeom << std::endl;
    for (int i = 0; i < scn.ngeom; i++) {
      geom = scn.geoms + i;
      if (geom->objid == bunny_id && geom->objtype == mjOBJ_GEOM)
        break;
    }
    uint32_t r = 254;
    uint32_t g = 0;
    uint32_t b = 255;
    geom->segid = (b << 16) | (g << 8) | r;
    // std::cout << geom->segid << std::endl;
    /*--------设置分割颜色--------*/
    mjr_render(viewport, &scn, &con);

    // swap OpenGL buffers (blocking call due to v-sync)
    glfwSwapBuffers(window);

    // process pending GUI events, call GLFW callbacks
    glfwPollEvents();
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
