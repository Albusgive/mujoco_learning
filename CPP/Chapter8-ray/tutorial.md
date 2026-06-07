# 射线检测 (Ray Casting)

射线检测（Ray Casting）常用于模拟激光雷达、超声波测距传感器，或者进行视线、碰撞预判。在仿真运行到 `mj_kinematics` 或 `mj_fwdPosition` 之后方可调用（因为射线检测依赖实体的实时位置信息）。

---

## 1. 核心 API 说明

### 1.1 多束射线检测 `mj_multiRay`
在一处起点发射多条不同方向的射线，常用于激光雷达雷达阵列：
```cpp
void mj_multiRay(const mjModel* m, mjData* d, const mjtNum pnt[3], const mjtNum* vec,
                 const mjtByte* geomgroup, mjtByte flg_static, int bodyexclude,
                 int* geomid, mjtNum* dist, int nray, mjtNum cutoff);
```

### 1.2 单条射线检测 `mj_ray`
检测单个射线的交点：
```cpp
mjtNum mj_ray(const mjModel* m, const mjData* d, const mjtNum pnt[3], const mjtNum vec[3],
              const mjtByte* geomgroup, mjtByte flg_static, int bodyexclude,
              int geomid[1]);
```

### 1.3 其他特定几何类型检测
* `mj_rayHfield`: 与高度图进行求交检测。
* `mj_rayMesh`: 与三角网格模型（Mesh）求交检测。
* `mju_rayGeom`: 用于纯几何计算（不受当前世界状态约束，计算指定位置尺寸的 geom）。

---

## 2. 关键参数详解

* **`pnt`** (`const mjtNum[3]`): 射线发射起点的 3D 笛卡尔坐标。
* **`vec`** (`const mjtNum*`): 射线方向向量。
  * 对于 `mj_multiRay`，其为长度为 `nray * 3` 的数组，每 3 个元素代表一条射线的方向向量。
* **`geomgroup`** (`const mjtByte*`): 几何体分组使能数组（长度为 `mjNGROUP`，通常为 5）。
  * 设为 `NULL` 代表检测所有 Geom 分组。若不为 `NULL`，则只有对应位为 1 的 Geom 组才会被检测。
* **`flg_static`** (`mjtByte`): 是否检测静态（Static）物体。`1` 代表检测静态物体，`0` 代表忽略静态物体。
* **`bodyexclude`** (`int`): 需要排除检测的 Body ID。
  * 设为 `-1` 代表检测所有 Body。
* **`geomid`** (`int*`): 传出参数。检测到相交的 Geom ID。若未击中任何物体，则返回 `-1`。
* **`dist`** (`mjtNum*`): 传出参数。到 Geom 表面的距离比例值。
  * **重要**：返回值不是绝对距离（单位米），而是**方向向量 `vec` 的缩放倍数**。即实际碰撞点坐标为：
    $$pos_{hit} = pnt + dist \times vec$$
  * 若未发生碰撞，返回值为 `-1`。
* **`nray`** (`int`): `mj_multiRay` 批量发射的射线总数。
* **`cutoff`** (`mjtNum`): 距离截断阈值。如果起点到几何体中心点的距离大于 `cutoff` 加上该几何体的包围球半径，则直接跳过检测以提升性能。

---

## 3. 官方文档入口
* [MuJoCo 官方 API 射线检测 (Ray Collisions)](https://mujoco.readthedocs.io/en/latest/programming.html#ray-collisions)