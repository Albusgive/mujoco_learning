# 仿真世界
## 世界根节点
```xml
<mujoco model="模型名称">
</mujoco>
```
## 仿真计算配置
### compiler节点
```xml
<compiler angle="radian/degree" autolimits="true" >
```
compiler节点中定义的包括angle（角度单位），autolimits（受力限制）等，常用的就像上面这样写就行，规定角度单位为弧度制，受力限制开启。角度单位为弧度制是机器人开发的常用单位制。
### option节点
```xml
<option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast" 
density="1.225" viscosity="1.8e-5"/>
```
* timestep代表仿真走一步的时间，也就是运行一次之后仿真计算出timestep时长后的世界，单位秒。timestep是一定要规定的，不然仿真不知道如何计算
* gravity 重力加速度
* wind 风在三个方向的速度
* magnetic="0 -0.5 0"世界磁场，影响磁力传感器
* density 介质密度，水，空气等，单位kg/m³
* viscosity 介质粘度
* **integrator** (`[Euler/RK4/implicit/implicitfast]`): 积分器，默认欧拉（Euler），用于每个仿真步长的物理求解。各自特点如下：
  * `Euler`: 简单快速，但精度低，适用于快速测试或简单系统。
  * `RK4`: 精度高，适用于对精度有要求但计算量不敏感的场景。
  * `Implicit`: 隐式积分，适合刚性系统（Stiff systems）和稳定性要求高的场景，但计算复杂度较高。
  * `ImplicitFast`: 在保证稳定性的前提下加快计算速度，适用于大规模复杂仿真。
* **solver** (`[PGS, CG, Newton]`): 约束求解器类型，默认牛顿法（Newton）。
* **iterations** (`"100"`): 约束求解器的最大迭代次数，按需配置。

#### 求解器细分配置参数 (Solver Attributes)
当需要对求解精度、性能进行更微观的调优时，可以在 `<option>` 中配置以下选项（均有其对应的官方默认值，大部分情况下无需手动配置）：

| 选项名称 | 类型与默认值 | 详细功能说明 | 官方文档参考 |
| :--- | :--- | :--- | :--- |
| `iterations` | `int, "100"` | 约束求解器的最大迭代次数。暖启动（Warmstart）开启时，能用较少迭代得到高精度解。 | [option-iterations](https://mujoco.readthedocs.io/en/latest/XMLreference.html#option-iterations) |
| `tolerance` | `real, "1e-8"` | 迭代求解器提前终止的容差阈值。当两次迭代的改进/梯度范数小于此值时自动停止。若设为 0 则禁用提前终止。 | [option-tolerance](https://mujoco.readthedocs.io/en/latest/XMLreference.html#option-tolerance) |
| `ls_iterations`| `int, "50"` | CG / Newton 求解器执行线搜索（Line Search）的最大迭代次数。 | [option-ls_iterations](https://mujoco.readthedocs.io/en/latest/XMLreference.html#option-ls_iterations) |
| `ls_tolerance` | `real, "0.01"` | 提前终止线搜索算法的容差阈值。 | [option-ls_tolerance](https://mujoco.readthedocs.io/en/latest/XMLreference.html#option-ls_tolerance) |
| `noslip_iterations`| `int, "0"` | Noslip 求解器的最大迭代次数（后处理步骤）。用于抑制软约束模型在摩擦维度上产生的多余滑移与漂移。设为 0 即禁用。 | [option-noslip_iterations](https://mujoco.readthedocs.io/en/latest/XMLreference.html#option-noslip_iterations) |
| `noslip_tolerance` | `real, "1e-6"` | 提前终止 Noslip 求解器的容差阈值。 | [option-noslip_tolerance](https://mujoco.readthedocs.io/en/latest/XMLreference.html#option-noslip_tolerance) |
| `ccd_iterations` | `int, "50"` | 凸面碰撞检测（CCD）算法的最大迭代次数。一般无需调整，除非几何体具有极大长宽比。 | [option-ccd_iterations](https://mujoco.readthedocs.io/en/latest/XMLreference.html#option-ccd_iterations) |
| `ccd_tolerance` | `real, "1e-6"` | 凸面碰撞算法提前终止的容差阈值。 | [option-ccd_tolerance](https://mujoco.readthedocs.io/en/latest/option-ccd_tolerance) |
| `sdf_iterations` | `int, "10"` | 有向距离场（SDF）碰撞的迭代次数（每个初始点）。 | [option-sdf_iterations](https://mujoco.readthedocs.io/en/latest/XMLreference.html#option-sdf_iterations) |
| `sdf_initpoints` | `int, "40"` | 寻找 Signed Distance Field 碰撞接触点时的起点数。 | [option-sdf_initpoints](https://mujoco.readthedocs.io/en/latest/XMLreference.html#option-sdf_initpoints) |

## 可视化配置
### visual节点
```xml
<visual>
    <global realtime="1"/>
    <quality shadowsize="16384" numslices="28" offsamples="4" />
    <headlight diffuse="1 1 1" specular="0.5 0.5 0.5" active="1" />
    <rgba fog="0 1 0 1" haze="1 0 0 1"/>
</visual>
```
* global:realtime仿真速度比例，在simulate中可以使用，大于1的按1计算
* quality:画面质量
* headlight：和simulate中自由相机相同方向的光源
* map：鼠标影响操作
* scale：渲染缩放
* rgba: fog 迷雾颜色；haze 地平线颜色

## 资源配置
```xml
<asset>
    <mesh name="tetrahedron" vertex="0 0 0 1 0 0 0 1 0 0 0 1" />
    <mesh file="card.obj" />
    <texture type="2d" file="./king_of_clubs.png" />
    <material name="king_of_clubs" texture="king_of_clubs" />
    <hfield name="agent_eval_gym" file="agent_eval_gym.png" size="10 10 1 1" />
    <texture type="skybox" file="../asset/desert.png"
        gridsize="3 4" gridlayout=".U..LFRB.D.." />
    <texture name="plane" type="2d" builtin="checker" rgb1=".1 .1 .1" rgb2=".9 .9 .9"
        width="512" height="512" mark="cross" markrgb=".8 .8 .8" />
    <material name="plane" reflectance="0.3" texture="plane" texrepeat="1 1" texuniform="true"/>
    <material name="box" rgba="0 0.5 0 1"  emission="0"/>
</asset>
```
### 几何资源 mesh
* vertex 通过坐标点构造几何体
* file 加载obj或者stl模型，不指定name默认为文件名
* hfield通过高度图加载几何体
### 纹理/材质资源
* texture 可以通过加载png对物体进行贴图，使用方法texture->material
* material 材质，可以指定纹理，颜色，反射，发光等
* skybox texture的类型，直接加载即可
**天空盒演示：**
```xml
<texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="0.6 0.8 1" width="256" height="256"/>
```