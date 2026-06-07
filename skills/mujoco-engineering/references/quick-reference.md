# MuJoCo Quick Reference & Essence (速查手册与精华提炼)

This document contains key formulas, API usage patterns, configuration references, and official documentation deep links. Use this for quick lookup during coding and explanation.

---

## 1. MJCF Modeling Quick Reference (建模速查)

### 1.1 Soft Contact (软接触与求解器)
MuJoCo simulates soft contacts using a dynamic "spring-damper" model.
* **Equation**: $a_{ref} = -b v - k r$ (where $r$ is penetration depth, $v$ is velocity, $b$ is damping, $k$ is stiffness).
* **solimp**: Calculates the constraint impedance $d(r) \in (0, 1)$.
  * Attributes: `(d0, dwidth, width, midpoint, power)`
  * Default: `(0.9, 0.95, 0.001, 0.5, 2)`
  * XML Syntax: `<geom solimp="0.9 0.95 0.001 0.5 2" .../>`
  * Official Reference: [MuJoCo option-solimp](https://mujoco.readthedocs.io/en/latest/XMLreference.html#option-solimp) | [Computation Contacts](https://mujoco.readthedocs.io/en/latest/computation/index.html#contacts)
* **solref**: Calculates stiffness $k$ and damping $b$.
  * Positive values: `(timeconst, dampratio)`. $b = \frac{2}{d_{width} \cdot timeconst}$, $k = \frac{d(r)}{d_{width} \cdot timeconst^2 \cdot dampratio^2}$.
  * Negative values: `(-stiffness, -damping)`. $b = \frac{damping}{d_{width}}$, $k = \frac{stiffness \cdot d(r)}{d_{width}^2}$.
  * XML Syntax: `<geom solref="0.02 1" .../>` or `<geom solref="-10000 -200" .../>`
  * Official Reference: [MuJoCo option-solref](https://mujoco.readthedocs.io/en/latest/XMLreference.html#option-solref)
* **Tuning Rules**:
  * For rubber/bouncy materials: set a larger `width` (e.g., 0.05) and adjust `solref` to control elasticity.
  * To prevent interpenetration (穿模): increase stiffness $k$ (by decreasing `timeconst` or increasing `-stiffness`). If it jitters, increase damping $b$.

### 1.2 Joints, Actuators & Sensors
* **Joints**: `<joint name="joint1" type="hinge" pos="0 0 0" axis="0 0 1" range="-90 90"/>`
  * Types: `hinge` (旋转), `slide` (滑动), `ball` (球形/三自由度), `free` (自由度/六自由度).
  * Official Reference: [MuJoCo joint](https://mujoco.readthedocs.io/en/latest/XMLreference.html#joint)
* **Actuators**:
  * Torque control: `<motor name="motor1" joint="joint1" gear="1"/>`
  * Position control: `<position name="pos1" joint="joint1" kp="100"/>`
  * Velocity control: `<velocity name="vel1" joint="joint1" kv="10"/>`
  * Official Reference: [MuJoCo actuator](https://mujoco.readthedocs.io/en/latest/XMLreference.html#actuator)
* **Sensors**:
  * IMU / Gyro / Accel: `<accelerometer name="acc1" site="site1"/>`, `<gyro name="gyro1" site="site1"/>`
  * Touch sensor: `<touch name="touch1" geom="geom1"/>`
  * Force / Torque: `<force name="force1" site="site1"/>`, `<torque name="torque1" site="site1"/>`
  * Official Reference: [MuJoCo sensor](https://mujoco.readthedocs.io/en/latest/XMLreference.html#sensor)

---

## 2. Python API Quick Reference (Python API)

### 2.1 Model Loading & Stepping (加载与步进)
```python
import mujoco

# Load model and create runtime data
model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)

# Simulation step
mujoco.mj_step(model, data)
```
* Official Reference: [MuJoCo Python Programming](https://mujoco.readthedocs.io/en/latest/programming.html#initialization)

### 2.2 Object Lookup (对象查名/ID)
```python
# Convert name to ID
geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "geom_name")
if geom_id == -1:
    raise ValueError("Geom not found")

# Get name from ID
geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
```
* Official Reference: [Indices and Names](https://mujoco.readthedocs.io/en/latest/programming.html#indices-and-names)

### 2.3 Sensor Reading (读取传感器)
* **New and Preferred Way (Attribute Access)**:
  ```python
  sensor_val = data.sensor("sensor_name").data
  ```
* **Classic Way (Index Access)**:
  ```python
  sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "sensor_name")
  start_idx = model.sensor_adr[sensor_id]
  dim = model.sensor_dim[sensor_id]
  sensor_val = data.sensordata[start_idx : start_idx + dim]
  ```
* Official Reference: [MuJoCo Sensor Programming](https://mujoco.readthedocs.io/en/latest/programming.html#sensors)

### 2.4 Offscreen Rendering & Camera Pixels (相机画面读取)
```python
import glfw
import numpy as np

# Initialize GLFW and hidden window
glfw.init()
glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
window = glfw.create_window(640, 480, "offscreen", None, None)
glfw.make_context_current(window)

# Setup renderer and camera
camera = mujoco.MjvCamera()
camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
camera.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "my_camera")

scene = mujoco.MjvScene(model, maxgeom=1000)
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, context)

# Render & read pixels
viewport = mujoco.MjrRect(0, 0, 640, 480)
mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
mujoco.mjr_render(viewport, scene, context)

rgb = np.zeros((480, 640, 3), dtype=np.uint8)
depth = np.zeros((480, 640), dtype=np.float32)
mujoco.mjr_readPixels(rgb, depth, viewport, context)
# Flip vertically because OpenGL coordinates start from bottom-left
rgb = np.flipud(rgb)
depth = np.flipud(depth)
```
* Official Reference: [MuJoCo Visualization Programming](https://mujoco.readthedocs.io/en/latest/programming.html#visualization)

---

## 3. C++ API & Compilation Quick Reference (C++ 编译与开发)

### 3.1 C++ Simulation Loop (基础C++仿真循环)
```cpp
#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>

mjModel* m = nullptr;
mjData* d = nullptr;

int main() {
    char error[1000];
    m = mj_loadXML("scene.xml", nullptr, error, 1000);
    if (!m) {
        mju_error("Could not load model: %s", error);
    }
    d = mj_makeData(m);

    // Main step loop
    while (!glfwWindowShouldClose(window)) {
        mj_step(m, d);
    }

    mj_deleteData(d);
    mj_deleteModel(m);
    return 0;
}
```

### 3.2 Minimum CMakeLists.txt (C++ 最小编译模板)
```cmake
cmake_minimum_required(VERSION 3.20)
project(mujoco_cpp_example)

# Option A: From CMake install directory (e.g., /opt/mujoco)
set(MUJOCO_FOLDER /opt/mujoco/lib/cmake)
find_package(mujoco REQUIRED PATHS ${MUJOCO_FOLDER} NO_DEFAULT_PATH)

add_executable(basic basic.cc)
target_link_libraries(basic mujoco::mujoco glut GL GLU glfw)
```
* Official Reference: [Building MuJoCo from source](https://mujoco.readthedocs.io/en/latest/programming.html#building-mujoco-from-source)

---

## 4. Common Pitfalls & Solutions (避坑指南)

* **GLIBCXX not found in Conda**:
  * *Error*: `ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version 'GLIBCXX_3.4.30' not found`
  * *Fix*: Run `conda install -c conda-forge libstdcxx-ng` inside the activated conda environment.
* **Header Include Order in C++**:
  * *Pitfall*: Including OpenGL or GLFW headers *before* MuJoCo headers can sometimes cause macro redefinition errors.
  * *Fix*: Always include `<mujoco/mujoco.h>` first.
* **GLFW Context in Headless Servers**:
  * *Pitfall*: Calling `glfw.create_window` on Linux headless servers without setting `glfw.window_hint(glfw.VISIBLE, glfw.FALSE)` or without X11 server will crash.
  * *Fix*: Use `os.environ["DISPLAY"]` checks or use EGL/OSMesa instead of GLFW if graphics driver is absent.

---

## 5. Interactive & Visual Teaching Templates (交互式与仿真式教学模板)

### 5.1 Standalone HTML Slider Visualizer (HTML 数学曲线交互分析模板)
Generate and save this template as an HTML file in the workspace or artifact directory when teaching curves like `solimp` or spring-damper equations:
```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>MuJoCo solimp 曲线交互可视化</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: sans-serif; margin: 20px; background: #f7f9fa; color: #333; }
        .container { max-width: 900px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .slider-group { margin: 15px 0; display: flex; align-items: center; }
        .slider-group label { width: 120px; font-weight: bold; }
        .slider-group input { flex-grow: 1; margin: 0 15px; }
        .slider-group span { width: 60px; text-align: right; }
    </style>
</head>
<body>
<div class="container">
    <h2>MuJoCo solimp (阻抗曲线) 交互分析器</h2>
    <p>公式: $d(r) = d_0 + Y(r/width) \cdot (d_{width} - d_0)$，调节下方参数滑块观察阻抗如何随穿模深度 $r$ 动态变化：</p>
    
    <div class="slider-group">
        <label>d0 (起始阻抗)</label>
        <input type="range" id="d0" min="0" max="1" step="0.01" value="0.9">
        <span id="d0-val">0.9</span>
    </div>
    <div class="slider-group">
        <label>dwidth (最大阻抗)</label>
        <input type="range" id="dwidth" min="0" max="1" step="0.01" value="0.95">
        <span id="dwidth-val">0.95</span>
    </div>
    <div class="slider-group">
        <label>width (归一化宽度)</label>
        <input type="range" id="width" min="0.0001" max="0.05" step="0.0001" value="0.001">
        <span id="width-val">0.001</span>
    </div>
    <div class="slider-group">
        <label>midpoint (分界点)</label>
        <input type="range" id="midpoint" min="0.01" max="0.99" step="0.01" value="0.5">
        <span id="midpoint-val">0.5</span>
    </div>
    <div class="slider-group">
        <label>power (曲线幂指数)</label>
        <input type="range" id="power" min="1" max="10" step="0.1" value="2">
        <span id="power-val">2</span>
    </div>

    <div id="chart" style="width:100%;height:450px;"></div>
</div>

<script>
    function computeY(x, mid, p) {
        if (x <= 0) return 0;
        if (x >= 1) return 1;
        let a = 1.0 / Math.pow(mid, p - 1);
        let b = 1.0 / Math.pow(1 - mid, p - 1);
        if (x <= mid) {
            return a * Math.pow(x, p);
        } else {
            return 1.0 - b * Math.pow(1 - x, p);
        }
    }

    function updatePlot() {
        const d0 = parseFloat(document.getElementById('d0').value);
        const dwidth = parseFloat(document.getElementById('dwidth').value);
        const w = parseFloat(document.getElementById('width').value);
        const mid = parseFloat(document.getElementById('midpoint').value);
        const p = parseFloat(document.getElementById('power').value);

        document.getElementById('d0-val').innerText = d0;
        document.getElementById('dwidth-val').innerText = dwidth;
        document.getElementById('width-val').innerText = w;
        document.getElementById('midpoint-val').innerText = mid;
        document.getElementById('power-val').innerText = p;

        let r_vals = [];
        let d_vals = [];
        // Compute curve up to 1.5 * width
        for (let r = 0; r <= w * 1.5; r += w / 100) {
            let x = r / w;
            let y = computeY(x, mid, p);
            let d = d0 + y * (dwidth - d0);
            r_vals.push(r);
            d_vals.push(d);
        }

        Plotly.newPlot('chart', [{
            x: r_vals,
            y: d_vals,
            mode: 'lines',
            line: { color: '#2b6cb0', width: 3 }
        }], {
            title: '阻抗值 d(r) 随穿模深度 r 变化曲线',
            xaxis: { title: '穿模深度 r (meter)' },
            yaxis: { title: '阻抗 d', range: [0, 1.05] }
        });
    }

    const inputs = document.querySelectorAll('input[type=range]');
    inputs.forEach(input => input.addEventListener('input', updatePlot));
    updatePlot();
</script>
</body>
</html>
```

### 5.2 Python Passive Viewer Keyboard Interaction (`launch_passive` 键盘交互仿真)
Write this script when teaching users how to interactively test control loops, apply forces, or tune friction dynamically inside MuJoCo's 3D viewer:
```python
import mujoco
import mujoco.viewer
import time
import sys

# Load model (make sure you have an actuator or control joint)
model = mujoco.MjModel.from_xml_path("API-MJCF/mecanum.xml")
data = mujoco.MjData(model)

# Active keyboard control loop
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("\n=== 交互控制台 ===")
    print("按下 [W] 增加控制目标，按下 [S] 减小控制目标")
    print("按下 [Q] 退出仿真")
    
    target_ctrl = 0.0

    while viewer.is_running():
        step_start = time.time()

        # Check key states (viewer.user_btn or custom key handlers if running window)
        # Note: In launch_passive, you can listen to terminal or use window events.
        # Alternatively, modify data.ctrl directly based on terminal triggers:
        # data.ctrl[0] = target_ctrl

        mujoco.mj_step(model, data)
        viewer.sync()

        # Step rate limiter
        time_elapsed = time.time() - step_start
        if time_elapsed < model.opt.timestep:
            time.sleep(model.opt.timestep - time_elapsed)
```

### 5.3 Custom 3D Debug Overlays (绘制自定义调试几何形状)
Use this boilerplate to teach users how to draw custom geometric objects (arrows, spheres, coordinate axes) dynamically in the 3D viewer without modifying the XML file:
```python
import mujoco
import mujoco.viewer
import numpy as np
import time

model = mujoco.MjModel.from_xml_path("API-MJCF/force.xml")
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()
        mujoco.mj_step(model, data)

        # Draw a custom 3D arrow to visualize contact force or direction
        # ngeoms should be reset or incremented on user_scn
        viewer.user_scn.ngeom = 0  # Clear previous custom shapes
        
        # Add a custom sphere at coordinates (0, 0, 1)
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[viewer.user_scn.ngeom],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.05, 0, 0],
            pos=[0.0, 0.0, 1.0],
            mat=np.eye(3).flatten(),
            rgba=[0, 1, 0, 0.8]  # Semi-transparent green
        )
        viewer.user_scn.ngeom += 1

        # Add a custom coordinate frame arrow pointing in Z-axis
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[viewer.user_scn.ngeom],
            type=mujoco.mjtGeom.mjGEOM_ARROW,
            size=[0.02, 0.02, 0.3], # width, thickness, length
            pos=[0.0, 0.0, 1.0],
            mat=np.eye(3).flatten(),
            rgba=[1, 0, 0, 1]  # Red arrow
        )
        viewer.user_scn.ngeom += 1

        viewer.sync()
        
        time_elapsed = time.time() - step_start
        if time_elapsed < model.opt.timestep:
            time.sleep(model.opt.timestep - time_elapsed)
```
