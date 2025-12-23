import time
import math

import mujoco
import mujoco.viewer

m = mujoco.MjModel.from_xml_path("../../API-MJCF/pointer.xml")
d = mujoco.MjData(m)

with mujoco.viewer.launch_passive(m, d) as viewer:
    # Close the viewer automatically after 30 wall-seconds.
    start = time.time()
    cnt = 0
    while viewer.is_running() and time.time() - start < 30:
        step_start = time.time()

        """测试step """
        d.ctrl[1] = math.sin(cnt)
        mujoco.mj_step(m, d)

        """测试step1 step2 """
        # mujoco.mj_step1(m, d)
        # d.ctrl[1] = math.sin(cnt)
        # mujoco.mj_step2(m, d)

        """测试forward """
        # d.ctrl[0] = math.sin(cnt)
        # d.qpos[0] = math.sin(cnt)
        # mujoco.mj_forward(m, d)
        # print("qvel:",d.qvel)
        # print("qacc:",d.qacc)
        # print("qpos:",d.qpos)

        """测试inverse """
        # d.qacc[0] = math.sin(cnt)
        # d.qpos[0] = 0
        # d.qvel[0] = 0
        # mujoco.mj_inverse(m, d)
        # print("qfrc_inverse",d.qfrc_inverse)

        cnt += 0.01

        # Example modification of a viewer option: toggle contact points every two seconds.
        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
