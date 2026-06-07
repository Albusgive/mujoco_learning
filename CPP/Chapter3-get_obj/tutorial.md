# 获取仿真世界中的实体信息 (Get Object Info)

在 MuJoCo 中，获取仿真世界中各种实体（Body, Joint, Geom 等）的数量、ID、位置及姿态是非常高频的操作。以下是实现逻辑与核心数据结构。

---

## 1. 获取实体数量与名称 (Get Counts & Names)
所有的数量与命名信息都存储在 `mjModel` 结构体中。

### 1.1 实体数量定义
在 `mjModel` 的开头，定义了所有仿真元素的数量（以 `n` 开头）：
```cpp
// 摘自 mujoco/mjmodel.h: 实体数量声明
int       nbody;                // 身体 (Body) 数量
int       njnt;                 // 关节 (Joint) 数量
int       ngeom;                // 几何体 (Geom) 数量
int       nsite;                // 标记点 (Site) 数量
int       ncam;                 // 相机 (Camera) 数量
int       nlight;               // 光源 (Light) 数量
int       nmesh;                // 网格 (Mesh) 数量
int       nsensor;              // 传感器 (Sensor) 数量
int       nactuator;            // 驱动器 (Actuator) 数量
```
* 源码链接：[mujoco/mjmodel.h (sizes段)](https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mjmodel.h)

### 1.2 实体名字缓冲区
在 `mjModel` 中，名字均以一维字符缓冲区存储，并通过对应的地址数组记录每个实体的名字指针偏置：
```cpp
// 摘自 mujoco/mjmodel.h: 命名缓冲区
char*     names;                // 名字缓存区，字符大数组 (nnames x 1)
int*      name_bodyadr;         // 各 Body 名字在 names 中的起始索引 (nbody x 1)
int*      name_jntadr;          // 各 Joint 名字在 names 中的起始索引 (njnt x 1)
int*      name_geomadr;         // 各 Geom 名字在 names 中的起始索引 (ngeom x 1)
int*      name_sensoradr;       // 各 Sensor 名字在 names 中的起始索引 (nsensor x 1)
```
* 源码链接：[mujoco/mjmodel.h (names段)](https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mjmodel.h)

---

## 2. 获取实体 ID (Get ID by Name)
因为 MuJoCo 底层物理计算全部基于数组索引（即 ID），在 API 中我们通常需要通过名字（string）获取其 ID。
使用 API 函数：
```cpp
MJAPI int mj_name2id(const mjModel* m, int type, const char* name);
```

### 2.1 参数说明
* `m`: `mjModel` 指针。
* `type`: 实体的类型，由枚举类型 `mjtObj` 定义。
* `name`: 实体的名称。

### 2.2 实体类型枚举 `mjtObj`
`mjtObj` 定义在 `mjmodel.h` 中，代表各种可能的实体类型：
```cpp
typedef enum mjtObj_ {
  mjOBJ_UNKNOWN        = 0,     // 未知对象类型
  mjOBJ_BODY,                   // 身体 (Body)
  mjOBJ_XBODY,                  // Body的替代表示，用于读取常规参考系 (而非惯性系)
  mjOBJ_JOINT,                  // 关节 (Joint)
  mjOBJ_DOF,                    // 自由度 (Degrees of freedom)
  mjOBJ_GEOM,                   // 几何体 (Geom)
  mjOBJ_SITE,                   // 标记点 (Site)
  mjOBJ_CAMERA,                 // 相机 (Camera)
  mjOBJ_LIGHT,                  // 光源 (Light)
  mjOBJ_FLEX,                   // 柔性可变形体 (Flex)
  mjOBJ_MESH,                   // 三角网格 (Mesh)
  mjOBJ_SKIN,                   // 皮肤网格 (Skin)
  mjOBJ_HFIELD,                 // 高度图 (Heightfield)
  mjOBJ_TEXTURE,                // 纹理 (Texture)
  mjOBJ_MATERIAL,               // 材质 (Material)
  mjOBJ_PAIR,                   // 特殊碰撞 geom 对 (Pair)
  mjOBJ_SENSOR                  // 传感器 (Sensor)
} mjtObj;
```
* 源码链接：[mujoco/mjmodel.h (mjtObj定义)](https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mjmodel.h)

演示代码：
```cpp
int body_id = mj_name2id(model, mjOBJ_BODY, "robot_link1");
```

---

## 3. 获取位置与姿态 (Get Pos & Ori)
物体的实时笛卡尔空间状态（三维位置、姿态矩阵等）全部存储在 **`mjData`** 中，由求解器在 `mj_step` 时进行实时更新更新：

```cpp
// 摘自 mujoco/mjdata.h: 笛卡尔坐标状态
mjtNum*   xpos;                 // 各个 Body 的三维笛卡尔中心坐标 (nbody x 3)
mjtNum*   xquat;                // 各个 Body 的笛卡尔四元数旋转姿态 (nbody x 4)
mjtNum*   xmat;                 // 各个 Body 的笛卡尔旋转矩阵姿态 (nbody x 9)
mjtNum*   xipos;                // 各个 Body 质心 (com) 的笛卡尔坐标 (nbody x 3)
mjtNum*   ximat;                // 各个 Body 质心的笛卡尔旋转矩阵 (nbody x 9)
```
* 源码链接：[mujoco/mjdata.h (Cartesian state)](https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mjdata.h)

演示代码：
```cpp
// 获取特定 body 实时三维位置坐标
int body_id = mj_name2id(model, mjOBJ_BODY, "support");
if (body_id != -1) {
    double x = data->xpos[body_id * 3 + 0];
    double y = data->xpos[body_id * 3 + 1];
    double z = data->xpos[body_id * 3 + 2];
    std::cout << "Body position: " << x << ", " << y << ", " << z << std::endl;
}
```
