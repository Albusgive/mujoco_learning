import time
import math

import mujoco
import mujoco.viewer
import cv2
import glfw
import numpy as np

m = mujoco.MjModel.from_xml_path('../../API-MJCF/deep_ray.xml')
d = mujoco.MjData(m)

camID = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "look_box")
cam_pos = m.cam_pos[camID*3:camID*3+3]

box_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "box1")
box2_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "box2")

box_num = 5
box_idx = np.zeros((box_num, 1), dtype=np.int32)
boxs_pos = np.zeros((box_num, 3))
for i in range(box_num):
  geom_name = "box" + str(i+1)
  box_idx[i] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
  boxs_pos[i] = d.geom_xpos[box_idx[i]]

with mujoco.viewer.launch_passive(m, d) as viewer:
  start = time.time()
  while viewer.is_running():
    
    step_start = time.time()
    mujoco.mj_step(m, d)

    '''--------单射线--------'''
    box_pos = d.geom_xpos[box_id]
    box2_pos = d.geom_xpos[box2_id]
    vec1 = box_pos - cam_pos
    vec2 = box2_pos - cam_pos
    vec1 = vec1.reshape(3, 1)
    vec2 = vec2.reshape(3, 1)
    geomid = np.zeros((1,1), dtype=np.int32)
    geomgroup = np.ones((6, 1))
    pnt = cam_pos.reshape(3, 1)
    x1 = mujoco.mj_ray(m,d,pnt,vec1,geomgroup,1,-1,geomid)
    x2 = mujoco.mj_ray(m,d,pnt,vec2,geomgroup,1,-1,geomid)
    distance1 = mujoco.mju_norm3(vec1) * x1
    distance2 = mujoco.mju_norm3(vec2) * x2
    # print(x1, x2)
    # print(distance1, distance2)
    '''--------单射线--------'''
    
    '''--------多射线--------'''
    for i in range(box_num):
      boxs_pos[i] = d.geom_xpos[box_idx[i]]
    num_vec = np.zeros((box_num*3,1))
    for i in range(box_num):
      for j in range(3):
        num_vec[i*3+j] = boxs_pos[i][j] - cam_pos[0][j]
    geomid = np.zeros((box_num,1), dtype=np.int32)
    normal = np.zeros((box_num*3,1), dtype=np.float64)
    geomgroup = np.ones((6, 1))
    dist = np.zeros((box_num,1))
    dist_ratio = np.zeros((box_num,1))
    pnt = cam_pos.reshape(3, 1)
    mujoco.mj_multiRay(m,d,pnt,num_vec,geomgroup,1,-1,geomid,dist_ratio,normal,box_num,999)
    for i in range(box_num):
      if geomid[i] == -1:
        dist[i] = -1.0
        continue
      dist[i] = mujoco.mju_norm3(num_vec[i*3:i*3+3]) * dist_ratio[i]
    print(dist)
    '''--------多射线--------'''


    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)