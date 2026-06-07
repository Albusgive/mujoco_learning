# 关节 (Joint)
***&emsp;&emsp;关节（Joint）用于连接相邻的 Body，并定义它们之间的相对自由度（DoF）***

在 MuJoCo 中，所有的几何体（Geom）都必须依附于身体（Body）。关节的作用是声明当前 Body 相对于其父级 Body 的运动方式。
* **运动主体**：关节对于其父级 Body 在物理上是相对静止的，当前 Body 则围绕/沿着关节定义的方向进行运动。
* **约束限制**：当前 Body 内部最多只能定义一组关节，用来表达它与上一层父级 Body 之间的连接关系。

---

## 1. 关节类型 (`type`)

MuJoCo 支持四种核心关节类型：

| 关节类型 | 英文名称 | 自由度 (DoF) | 运动描述 | 适用场景 |
| :--- | :--- | :---: | :--- | :--- |
| **旋转关节** | `hinge` | 1 | 围绕指定的 `axis` 进行单轴旋转。 | 摆臂、轮轴、人体关节。 |
| **滑动关节** | `slide` | 1 | 沿着指定的 `axis` 进行单轴平移。 | 电梯、活塞、伸缩机构。 |
| **球关节** | `ball` | 3 | 绕关节中心进行任意三维旋转（类似于球头万向节）。 | 肩关节、摇杆。 |
| **自由关节** | `free` | 6 | 包含 3 维平移与 3 维旋转，允许物体在空间中完全自由运动。 | 漂浮物、飞行的抛体。*(仅能声明在根 Body 下)* |

---

## 2. 常用 XML 属性说明

编写 MJCF 模型文件时，`<joint>` 标签的常用属性配置如下：

| 属性名称 | 默认值 | 说明与计算公式 | 官方文档跳转 |
| :--- | :--- | :--- | :--- |
| `name` | `""` | 关节的唯一名称，用于在 Python/C++ API 中进行 ID 寻址。 | [joint-name](https://mujoco.readthedocs.io/en/latest/XMLreference.html#joint-name) |
| `type` | `hinge` | 关节类型，可选 `hinge`, `slide`, `ball`, `free`。 | [joint-type](https://mujoco.readthedocs.io/en/latest/XMLreference.html#joint-type) |
| `pos` | `0 0 0` | 关节锚点在**当前 Body 坐标系**下的三维坐标。 | [joint-pos](https://mujoco.readthedocs.io/en/latest/XMLreference.html#joint-pos) |
| `axis` | `0 0 1` | 运动轴线的矢量方向（仅对 `hinge` 和 `slide` 有效）。 | [joint-axis](https://mujoco.readthedocs.io/en/latest/XMLreference.html#joint-axis) |
| `stiffness`| `0` | 关节弹簧刚度系数 $k$。产生的回弹力公式为：<br> $f_{stiffness} = (ref - qpos) \times stiffness$ | [joint-stiffness](https://mujoco.readthedocs.io/en/latest/XMLreference.html#joint-stiffness) |
| `damping` | `0` | 关节扭转阻尼系数 $c$。产生的阻尼力公式为：<br> $f_{damping} = (0 - qvel) \times damping$ | [joint-damping](https://mujoco.readthedocs.io/en/latest/XMLreference.html#joint-damping) |
| `frictionloss`| `0` | 关节干摩擦损失限额。用于模拟关节静摩擦与库伦摩擦阻力。 | [joint-frictionloss](https://mujoco.readthedocs.io/en/latest/XMLreference.html#joint-frictionloss) |
| `armature` | `0` | 电枢惯量（转子转动惯量 $\times$ 减速比$^2$）。用于模拟电机转子的惯性。 | [joint-armature](https://mujoco.readthedocs.io/en/latest/XMLreference.html#joint-armature) |
| `ref` | `0` | 关节弹簧的平衡位置偏置量（角度值或平移量）。 | [joint-ref](https://mujoco.readthedocs.io/en/latest/XMLreference.html#joint-ref) |
| `range` | `0 0` | 运动范围限制。例如转动关节 `range="-1.57 1.57"` (单位为弧度或度)。 | [joint-range](https://mujoco.readthedocs.io/en/latest/XMLreference.html#joint-range) |
| `limited` | `auto` | 是否启用运动限位。设置为 `true` 启用，`false` 禁用，`auto` 自动（依据 range 自动决定）。 | [joint-limited](https://mujoco.readthedocs.io/en/latest/XMLreference.html#joint-limited) |

---

## 3. 经典动力学公式解析

MuJoCo 在仿真物理步进时，会将关节的弹簧、阻尼、电枢惯量等物理效应自动折算进**被动力**（`qfrc_passive`）中进行统一求解：

### 3.1 弹簧恢复力 (Stiffness Force)
弹簧力拉引关节回到参考点 `ref`。
$$f_{stiffness} = - \text{stiffness} \times (qpos - \text{ref})$$

### 3.2 阻抗力 (Damping Force)
阻尼力用于阻碍关节相对运动，稳定振荡。
$$f_{damping} = - \text{damping} \times qvel$$

### 3.3 关节摩擦损失 (Friction Loss)
关节摩擦力属于非平滑约束，它由引擎的约束解算器（Constraint Solver）计算，以防止关节微小的滑动趋势。具体计算过程会体现在约束力 `efc_force` 中。

---

## 4. 实例分析

以下是本章示例模型 [scence.xml](file:///home/albusgive2/mujoco_learning/MJCF/Chapter4-joint/scence.xml) 中的双关节悬挂倒立摆场景定义：

```xml
<!-- 支撑柱 -->
<body name="support" pos="0 0 0.1">
    <geom type="cylinder" mass="100" size="0.05 0.5" rgba="0.2 0.2 0.2 1"/>
    
    <!-- 水平旋转杆 -->
    <body name="rotay_am" pos="0 0 0.51">
        <!-- 关节1：绕Z轴旋转的Hinge关节，带阻尼与弹簧回复力 -->
        <joint type="hinge" name="pivot" pos="0 0 0" axis="0 0 1" damping="0.001" frictionloss="0.0" stiffness="0.5"/>
        <geom type="capsule" mass="0.01" fromto="0 0 0 0.2 0 0" size="0.01" rgba="0.8 0.2 0.2 0.5"/>
        
        <!-- 挂载在水平杆末端的自由下摆 -->
        <body name="pendulum" pos="0.2 0 0">
            <!-- 关节2：绕X轴自由旋转的Hinge摆关节，无弹簧刚度 -->
            <joint type="hinge" name="ph" pos="0 0 0" axis="1 0 0" damping="0.001" frictionloss="0.0" />
            <geom type="capsule" mass="0.005" fromto="0 0 0 0 0 -0.3" size="0.01" rgba="0.8 0.2 0.2 1" />
            <!-- 末端配重 -->
            <geom type="sphere" mass="0.01" size="0.03" pos="0 0 -0.3" rgba="0.2 0.8 0.2 1" />
        </body>
    </body>
</body>
```

* **水平转轴 `pivot`**：沿着 `axis="0 0 1"`（Z轴），配置了 `stiffness="0.5"`。只要它发生旋转偏移，弹簧就会产生扭矩强行拉回初始朝向。
* **摆动轴 `ph`**：沿着 `axis="1 0 0"`（X轴），配置了阻尼 `damping="0.001"` 以平抑剧烈振荡，但没有设置刚度，允许倒立摆像真实单摆一样下垂和自由摆动。

---

## 5. 官方文档入口
关于关节在 MuJoCo 中的计算原理与全部 XML 配置项，请参考：
* [MuJoCo 官方 XML 关节参考 (Joint XML Reference)](https://mujoco.readthedocs.io/en/latest/XMLreference.html#joint)
* [MuJoCo 官方计算与约束理论 (Computation Constraints)](https://mujoco.readthedocs.io/en/latest/computation/index.html#constraints)