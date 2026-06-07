import time

import mujoco
import mujoco.viewer
import math

import pygame
pygame.mixer.init()
pygame.mixer.music.load('Assumptions.mp3')


m = mujoco.MjModel.from_xml_path('scence.xml')
d = mujoco.MjData(m)

with mujoco.viewer.launch_passive(m, d) as viewer:
#   time.sleep(10)
  pygame.mixer.music.play()
  start = d.time
  max_ctrl = 1.0
  ctrl_step = 0.5
  ctrl_time = 2
  ctrl_time_step = 0.1
  while viewer.is_running():
    step_start = time.time()

    if d.time > 10:
        if d.time - start > ctrl_time:
            start = d.time
            #渐入佳境
            ctrl_time -= ctrl_time_step
            if ctrl_time < 0.46:#44拍
                ctrl_time = 0.46
            
            #绝对值
            if abs(max_ctrl) < 10:
                if max_ctrl > 0:
                    max_ctrl += ctrl_step
                else:
                    max_ctrl -= ctrl_step
            max_ctrl = -max_ctrl
        
        d.ctrl[0] = -max_ctrl
        d.ctrl[1] = max_ctrl
    
    if d.time > 20:
        d.ctrl[2] = 0.6

    
      
    mujoco.mj_step(m, d)


    viewer.sync()

    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)