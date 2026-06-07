# 作用力
**平动**      

$$ m \cdot a = F $$

**旋转**

$$ I \cdot \alpha = \tau $$  

$$ F/\tau = 外部力+驱动力+被动力+约束力+偏置力 $$  

# 外部力
我们查看文档计算部分可以看到有qfrc_passive,qfrc_actuator,qfrc_applied三个力分别对应被动力，驱动力，外部力       

```c
mjtNum* qfrc_applied;      // applied generalized force             (nv x 1)
mjtNum* qfrc_passive;      // passive generalized force             (nv x 1)
mjtNum* qfrc_actuator;     // actuator generalized force            (nv x 1)
```

只要看手册的api或者头文件中，找到mj_applyFT函数应用外部力。

```c
void mj_applyFT(const mjModel* m, mjData* d,
                const mjtNum force[3], const mjtNum torque[3],
                const mjtNum point[3], int body,
                mjtNum* qfrc_target);
```

或者还可以使用xfrc_applied直接作用外部力在质心上。

```c
mjtNum* xfrc_applied;      // applied Cartesian forces              (nbody x 6)
```

**这里也说明了是笛卡尔力。**
&emsp;&emsp;mj_applyFT函数的参数，是三维的力，三维扭矩，三维坐标(worldbody坐标系)，bodyid。qfrc_target可以直接使用d->qfrc_applied。mj_applyFT是对于body在 **“自由度”** 上施加力。于是我们可以使用两个方式对body施加外部力。       
&emsp;&emsp;qfrc_target还可以是以下这些被动力等qfrc_xxx的力

```c
mjtNum* qfrc_passive;      // passive generalized force             (nv x 1)
mjtNum* qfrc_bias;         // constraint bias generalized force     (nv x 1)
mjtNum* qfrc_constraint;   // constraint generalized force          (nv x 1)
```

**mj_Data接口演示（作用在质心上）：**
```C++
int bullet_id = mj_name2id(m, mjOBJ_BODY, "box");
mjtNum *set_torque = d->xfrc_applied + bullet_id * 6;
```

**mj_applyFT函数接口演示（可以调整施力点）:**
```C++
int bullet_id = mj_name2id(m, mjOBJ_BODY, "box");
mjtNum force[3] = {0.0, 0.0, 9.81};
mjtNum torque[3] = {0.0, 0.0, 0.0};
mjtNum point[3] = {0.0, 0.0, 0.0}; 
mj_applyFT(m, d, force, torque, point, id, d->qfrc_applied);
```
**mj_applyFT每次调用都是增量式，如果我们想清除力可以使用mju_zero，如mju_zero(d->qfrc_applied, m->nv);**

# 驱动力
```c
mjtNum* qfrc_actuator;     // actuator generalized force            (nv x 1)
```
&emsp;&emsp;mjData.qfrc_actuator是驱动器执行的力，不同驱动器最终会计算出力或者扭矩作用到关节上。        

# 被动力
&emsp;&emsp;mjData.qfrc_passive是被动力，关节参数的damping，stiffness,摩擦力，流体阻力都会最终计算到改力中。        

$$ damping_force = (0-qvel)*damping $$

$$ stiffness_force = (0-qpos)*stiffness $$

# 约束力
```c
mjtNum* efc_force;         // constraint force                      (nefc x 1)
```
&emsp;&emsp;mjData.efc_force是约束力，关节的frictionloss，equality计算出来的合力为改力。        
&emsp;&emsp;这里 $jar_0 = Jac \cdot qacc_0 - aref$ 表示无约束时的约束空间残差（未施加摩擦力时的相对加速度偏离量），$InverseConstraintMass$ 是该约束维度的质量倒数（即惯量响应矩阵 $A = J M^{-1} J^T + R$ 的对应对角元素）。

关节干摩擦力（静摩擦与滑动摩擦）的完整解算公式为：

$$
frictionloss\_force = 
\begin{cases}
    frictionloss, & \text{if } jar_0 \le -InverseConstraintMass \cdot floss  & \text{(正向滑动摩擦)} \\
    -frictionloss,  & \text{else if } jar_0 \ge InverseConstraintMass \cdot floss   & \text{(反向滑动摩擦)} \\
    -\frac{jar_0}{InverseConstraintMass},  & \text{else } & \text{(静摩擦/粘滞状态，此时实际残差 } jar = 0\text{)}
\end{cases}
$$

**原理解析**：
1. **滑动摩擦阶段**：当外力推动关节的趋势（即 $jar_0$）超过了最大静摩擦力所能提供的阻碍加速度时，摩擦力饱和，大小恒为最大摩擦力 $floss$（即 `frictionloss`），方向与相对运动趋势相反。
2. **静摩擦（粘滞）阶段**：当外力较小，在摩擦力能平衡的范围内（即 `else` 分支），摩擦力会自动产生一个恰好抵消相对运动趋势的力 $-\frac{jar_0}{InverseConstraintMass}$，使得最终的实际加速度残差 $jar$ 完美归零，即物体保持静止或相对匀速运动。
**源码实现(SRC/engine/engine_core_constraint.c)**
```c
// linear friction limits
else if (floss && floss[i] > 0) {
  // linear negative friction (正向滑动摩擦)
  if (jar[i] <= -R[i] * floss[i]) {
    force[i] = floss[i];
    state[i] = mjCNSTRSTATE_LINEARNEG;
  }

  // linear positive friction (反向滑动摩擦)
  else if (jar[i] >= R[i] * floss[i]) {
    force[i] = -floss[i];
    state[i] = mjCNSTRSTATE_LINEARPOS;
  }

  // quadratic/inactive friction (静摩擦/粘滞状态)
  else {
    force[i] = -jar[i] / R[i];
    state[i] = mjCNSTRSTATE_QUADRATIC;
  }
}
```

# 偏置力
```c
mjtNum* qfrc_bias;         // constraint bias generalized force     (nv x 1)
```
&emsp;&emsp;mjData.qfrc_bias科里奥利力等，由引擎自动计算。        