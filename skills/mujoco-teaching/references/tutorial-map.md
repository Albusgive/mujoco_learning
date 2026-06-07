# MuJoCo Tutorial Map

Use repository-relative paths only. Do not invent absolute paths.

## Core Entry Points

- `README.md`: project overview, local docs site, installation commands.
- `directory.md`: human-readable tutorial table of contents.
- `MJCF/Chapter0-install/tutorial.md`: installing MuJoCo from source/release/Python.

## MJCF Modeling Lessons

- World and visualization setup: `MJCF/Chapter2-virtual_world/tutorial.md`
- Bodies, geoms, sites, coordinate tree: `MJCF/Chapter3-worldbody/tutorial.md`
- Joints and joint parameters: `MJCF/Chapter4-joint/tutorial.md`
- Friction and contact dimensions: `MJCF/Chapter5-friction/tutorial.md`
- Actuators and control types: `MJCF/Chapter6-actuator/tutorial.md`
- Lights, cameras, replicate, ray demos: `MJCF/Chapter7-light&replicate/tutorial.md`
- Tendons and muscle-like actuation: `MJCF/Chapter8-tendon/tutorial.md`
- Sensors, camera, IMU, velocity, angles: `MJCF/Chapter9-sensor/tutorial.md`
- CAD model import workflow: `MJCF/Chapter10-from_CAD_software/tutorial.md`
- Equality constraints and parallel mechanisms: `MJCF/Chapter11-equality/tutorial.md`
- Defaults/classes/inheritance: `MJCF/Chapter12-default/tutorial.md`
- Old composite deformables: `MJCF/Chapter13-composite/tutorial.md`
- New flex deformables: `MJCF/Chapter14-flex/tutorial.md`
- Keyframes: `MJCF/Chapter15-keyframe/tutorial.md`

## Python API Lessons

- Viewer, model/data, stepping: `Python/Chapter1-view&step/tutorial.md`; code: `Python/Chapter1-view&step/view.py`
- Object/entity lookup: `Python/Chapter2-get_obj/tutorial.md`; code: `Python/Chapter2-get_obj/get_obj.py`
- Sensor data access: `Python/Chapter3-sensor_data/tutorial.md`; code: `Python/Chapter3-sensor_data/sensor_data.py`
- 2D/3D drawing: `Python/Chapter4-draw/tutorial.md`; code: `Python/Chapter4-draw/draw.py`
- Force terms and validation: `Python/Chapter5-force/tutorial.md`; code: `Python/Chapter5-force/force.py`
- Rendering configuration, segmentation, stereo: `Python/Chapter6-vis_cfg/tutorial.md`; code: `Python/Chapter6-vis_cfg/vis_cfg.py`
- Ray distance queries: `Python/Chapter7-ray/tutorial.md`; code: `Python/Chapter7-ray/ray.py`

## C++ API Lessons

- Build and CMake: `CPP/Chapter1-make/tutorial.md`; code: `CPP/Chapter1-make/`
- Viewer and stepping: `CPP/Chapter2-view&step/tutorial.md`; code: `CPP/Chapter2-view&step/basic.cc`
- Object/entity lookup: `CPP/Chapter3-get_obj/tutorial.md`; code: `CPP/Chapter3-get_obj/get_obj.cc`
- Sensor data access: `CPP/Chapter4-sensor_data/tutorial.md`; code: `CPP/Chapter4-sensor_data/sensor_data.cc`
- 2D/3D drawing: `CPP/Chapter5-draw/tutorial.md`; code: `CPP/Chapter5-draw/draw.cpp`
- Force terms and validation: `CPP/Chapter6-force/tutorial.md`; code: `CPP/Chapter6-force/force.cpp`
- Rendering configuration: `CPP/Chapter7-vis_cfg/tutorial.md`; code: `CPP/Chapter7-vis_cfg/vis_cfg.cpp`
- Ray distance queries: `CPP/Chapter8-ray/tutorial.md`; code: `CPP/Chapter8-ray/ray.cpp`

## Extensions And Special Topics

- Touch sensing: `extend/touch/readme.md`
- Soft contact and solver parameter intuition: `extend/soft_contact/tutorial.md`
- Ray caster / depth/range sensing: `extend/deep_camera/readme.md`; external repo: `https://github.com/Albusgive/mujoco_ray_caster`
- Equality experiments: `extend/equality/`
- JAX/MJX examples: `extend/jax/`
- Entertainment/redstone demo: `fun/mujoco_red_stone/README.md`

## Official Docs Fallback

Use official MuJoCo docs when local material is incomplete:

- Programming guide and source build: `https://mujoco.readthedocs.io/en/latest/programming/#building-mujoco-from-source`
- XML reference, API reference, and computation chapters from the same documentation site as needed.
