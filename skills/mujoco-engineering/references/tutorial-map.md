# MuJoCo Engineering Tutorial Map

Use repository-relative paths only. Inspect referenced files before coding.

## Project Setup And Installation

- Editable Python package and docs site: `README.md`, `pyproject.toml`, `setup.py`, `mujoco_learning_doc/`
- MuJoCo install/source build: `MJCF/Chapter0-install/tutorial.md`
- C++ build patterns: `CPP/Chapter1-make/CMakeLists.txt`, `CPP/Chapter1-make/tutorial.md`

## MJCF Authoring Targets

- Base world/options/assets/materials: `MJCF/Chapter2-virtual_world/tutorial.md`
- Body tree, geoms, sites, meshes, coordinate frames: `MJCF/Chapter3-worldbody/tutorial.md`
- Joints: `MJCF/Chapter4-joint/tutorial.md`
- Friction/contact parameters: `MJCF/Chapter5-friction/tutorial.md`
- Actuators: `MJCF/Chapter6-actuator/tutorial.md`
- Light/camera/replicate: `MJCF/Chapter7-light&replicate/tutorial.md`
- Tendons: `MJCF/Chapter8-tendon/tutorial.md`
- Sensors: `MJCF/Chapter9-sensor/tutorial.md`
- CAD import scripts/data: `MJCF/Chapter10-from_CAD_software/`
- Equality constraints: `MJCF/Chapter11-equality/tutorial.md`, `extend/equality/`
- Defaults/classes: `MJCF/Chapter12-default/tutorial.md`
- Composite/flex deformables: `MJCF/Chapter13-composite/tutorial.md`, `MJCF/Chapter14-flex/tutorial.md`
- Keyframes: `MJCF/Chapter15-keyframe/tutorial.md`

## Python API Examples

- Viewer/stepping: `Python/Chapter1-view&step/view.py`
- Object IDs and entity lookup: `Python/Chapter2-get_obj/get_obj.py`
- Sensor reads: `Python/Chapter3-sensor_data/sensor_data.py`
- Drawing/debug visualization: `Python/Chapter4-draw/draw.py`
- Force terms: `Python/Chapter5-force/force.py`
- Rendering configuration/segmentation/stereo: `Python/Chapter6-vis_cfg/vis_cfg.py`
- Ray queries: `Python/Chapter7-ray/ray.py`

## C++ API Examples

- Viewer/stepping: `CPP/Chapter2-view&step/basic.cc`
- Object IDs and entity lookup: `CPP/Chapter3-get_obj/get_obj.cc`
- Sensor reads: `CPP/Chapter4-sensor_data/sensor_data.cc`
- Drawing/debug visualization: `CPP/Chapter5-draw/draw.cpp`
- Force terms: `CPP/Chapter6-force/force.cpp`
- Rendering configuration: `CPP/Chapter7-vis_cfg/vis_cfg.cpp`
- Ray queries: `CPP/Chapter8-ray/ray.cpp`
- Shared helper: `utils/mujoco_thread/`

## Extension Examples

- Touch sensing: `extend/touch/readme.md`, `extend/touch/C++/`
- Soft contact: `extend/soft_contact/tutorial.md`, `extend/soft_contact/soft_contact.py`, `extend/soft_contact/C++/soft_contact.cpp`, related XML files in `extend/soft_contact/`
- Ray caster / depth/range sensing: `extend/deep_camera/readme.md`, `extend/deep_camera/C++/RayCasterCamera.hpp`, external repo `https://github.com/Albusgive/mujoco_ray_caster`
- MJX/JAX experiments: `extend/jax/`
- Entertainment/redstone demo: `fun/mujoco_red_stone/`

## Validation Commands

- Python syntax for docs web: `conda run -n mj python -m compileall mujoco_learning_doc`
- Install package in target environment: `python -m pip install -e .`
- Start docs site: `mujoco_learning_doc --port 8000`
- For C++ chapters, follow each chapter's `CMakeLists.txt`; avoid reusing generated `build/`, `b/`, or cache artifacts as source.

## Minimal Demo And Validation Assets

Use these repository-relative assets selectively when parameter tuning, reproduction, or validation keeps failing and a smaller isolated demo would clarify the issue:

- General XML examples: `API-MJCF/`, especially `mecanum.xml`, `force.xml`, `vis_cfg.xml`, `deep_ray.xml`
- MJCF chapter scene files: `MJCF/Chapter*/`
- Python chapter scripts: `Python/Chapter*/`
- C++ chapter examples and `CMakeLists.txt`: `CPP/Chapter*/`
- Soft contact XML/Python/C++ examples: `extend/soft_contact/`
- Ray caster examples: `extend/deep_camera/`, `Python/Chapter7-ray/`, `CPP/Chapter8-ray/`
- Touch/equality/JAX extension examples: `extend/touch/`, `extend/equality/`, `extend/jax/`

For multimodal agents, a diagnostic demo may include screenshots or video captured from the viewer/render pipeline and compared with the expected visual effect. For text-only agents, produce deterministic textual outputs from simulation state, such as sensor arrays, contact counts, `qpos`, `qvel`, forces, ray hit distances, or rendered image shape/statistics.

## Official Docs Fallback

Use official MuJoCo docs when local examples do not answer an API or build detail:

- Programming/source build: `https://mujoco.readthedocs.io/en/latest/programming/#building-mujoco-from-source`
- Use the XML/API/reference sections on the same site for exact attribute and function semantics.
