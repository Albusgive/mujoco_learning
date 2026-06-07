### MJCF模型文件
> 第一节   mujoco安装 [install](MJCF/Chapter0-install/tutorial.md) | [官方文档: 安装与构建](https://mujoco.readthedocs.io/en/latest/programming.html#building-mujoco-from-source)  
> 第二节   仿真世界调整 [virtual_world](MJCF/Chapter2-virtual_world/tutorial.md) | [官方文档: option/visual/asset](https://mujoco.readthedocs.io/en/latest/XMLreference.html#option)  
> - 物理世界参数，可视化配置，材质加载等
> 
> 第三节   仿真世界 [worldbody](MJCF/Chapter3-worldbody/tutorial.md) | [官方文档: worldbody/body/geom](https://mujoco.readthedocs.io/en/latest/XMLreference.html#worldbody)  
> - 世界body，几何体，地形等
>
> 第四节   关节 [joint](MJCF/Chapter4-joint/tutorial.md) | [官方文档: joint](https://mujoco.readthedocs.io/en/latest/XMLreference.html#joint)  
> - 关节类型，关节动力，关节参数等
>
> 第五节   摩擦力设置及计算方式 [friction](MJCF/Chapter5-friction/tutorial.md) | [官方文档: friction/contact](https://mujoco.readthedocs.io/en/latest/XMLreference.html#pair)  
> - mujoco中多维度的摩擦力调整
> 
> 第六节   驱动器 [actuator](MJCF/Chapter6-actuator/tutorial.md) | [官方文档: actuator](https://mujoco.readthedocs.io/en/latest/XMLreference.html#actuator)  
> - 添加速度控制，位置控制，力矩控制等
>
> 第七节   灯光和复制 [light&replicate](MJCF/Chapter7-light&replicate/tutorial.md) | [官方文档: light/camera](https://mujoco.readthedocs.io/en/latest/XMLreference.html#light)  
> - 灯光类型(和相机传感器关联性强)
> - 复制实体，阵列排布，激光雷达演示
>
> 第八节   肌腱 [tendon](MJCF/Chapter8-tendon/tutorial.md) | [官方文档: tendon](https://mujoco.readthedocs.io/en/latest/XMLreference.html#tendon)  
> - mujoco特有的驱动器和关节联系方式
> - 肌肉控制
>
> 第九节   传感器 [sensor](MJCF/Chapter9-sensor/tutorial.md) | [官方文档: sensor](https://mujoco.readthedocs.io/en/latest/XMLreference.html#sensor)  
> - 相机传感器，imu,速度，角度等
>
> 第十节   从CAD软件制作mjcf模型 [from_CAD_software](MJCF/Chapter10-from_CAD_software/tutorial.md) | [官方文档: overview](https://mujoco.readthedocs.io/en/latest/overview.html#model-description)  
> - 以solidworks为例
>
> 第十一节   约束条件 [equality](MJCF/Chapter11-equality/tutorial.md) | [官方文档: equality](https://mujoco.readthedocs.io/en/latest/XMLreference.html#equality)  
> - 并联机构建模，驱动跟踪等
>
> 第十二节   默认属性设置 [default](MJCF/Chapter12-default/tutorial.md) | [官方文档: default](https://mujoco.readthedocs.io/en/latest/XMLreference.html#default)  
> - 几何体，body，关节等默认参数
>
> 第十三节   可变形体（老版3.2.7及以前） [composite](MJCF/Chapter13-composite/tutorial.md) | [官方文档: composite](https://mujoco.readthedocs.io/en/latest/XMLreference.html#composite)  
> - 绳子，布料，软体等
>
> 第十四节   可变形体（新版3.3.0及以后） [flex](MJCF/Chapter14-flex/tutorial.md) | [官方文档: flex](https://mujoco.readthedocs.io/en/latest/XMLreference.html#flex)  
> - 柔性材料，布料，软体，绳索，从网格构建可变形模型等
>
> 第十五节   关节帧 [keyframe](MJCF/Chapter15-keyframe/tutorial.md) | [官方文档: keyframe](https://mujoco.readthedocs.io/en/latest/XMLreference.html#keyframe)  
> - 加载和储存特定的姿态，关节信息等
### API
> 第一节   编译 
> - 编译环境，编译命令，编译开发演示
> > [make(C++)](CPP/Chapter1-make/tutorial.md) | [官方文档: 编译与链接](https://mujoco.readthedocs.io/en/latest/programming.html#building-mujoco-from-source)
>
> 第二节   可视化和仿真进行 
> - 仿真环境，可视化，仿真步进，仿真步进控制等
> > [view&step(C++)](CPP/Chapter2-view&step/tutorial.md) | [官方文档: 引擎初始化](https://mujoco.readthedocs.io/en/latest/programming.html#initialization)
> > [view&step(Python)](Python/Chapter1-view&step/tutorial.md) | [官方文档: 引擎初始化](https://mujoco.readthedocs.io/en/latest/programming.html#initialization)
>
> 第三节   获取仿真世界中的实体信息 
> - 获取仿真世界中的实体信息，如名字，数量，参数信息等
> > [get_obj(C++)](CPP/Chapter3-get_obj/tutorial.md) | [官方文档: 实体索引与名称](https://mujoco.readthedocs.io/en/latest/programming.html#indices-and-names)
> > [get_obj(Python)](Python/Chapter2-get_obj/tutorial.md) | [官方文档: 实体索引与名称](https://mujoco.readthedocs.io/en/latest/programming.html#indices-and-names)
>
> 第四节   传感器数据获取 
> - 获取仿真世界中的传感器数据，如相机，imu，速度，角度等
> > [sensor_data(C++)](CPP/Chapter4-sensor_data/tutorial.md) | [官方文档: 传感器读取](https://mujoco.readthedocs.io/en/latest/programming.html#sensors)
> > [sensor_data(Python)](Python/Chapter3-sensor_data/tutorial.md) | [官方文档: 传感器读取](https://mujoco.readthedocs.io/en/latest/programming.html#sensors)
>
> 第五节   2D和3D绘制 
> - 2D绘制：文字，方形，表格等
> - 3D绘制：基础几何体，箭头等
> > [draw(C++)](CPP/Chapter5-draw/tutorial.md) | [官方文档: 自定义绘制](https://mujoco.readthedocs.io/en/latest/programming.html#visualization)
> > [draw(Python)](Python/Chapter4-draw/tutorial.md) | [官方文档: 自定义绘制](https://mujoco.readthedocs.io/en/latest/programming.html#visualization)
>
> 第六节   力的计算和API验证 
> - mujoco中力是如何作用的及验证
> > [force(C++)](CPP/Chapter6-force/tutorial.md) | [官方文档: 动力学状态](https://mujoco.readthedocs.io/en/latest/programming.html#physics-state)
> > [force(Python)](Python/Chapter5-force/tutorial.md) | [官方文档: 动力学状态](https://mujoco.readthedocs.io/en/latest/programming.html#physics-state)
>
> 第七节   渲染配置 
> - 双目相机，图像分割，渲染配置等
> > [vis_cfg(C++)](CPP/Chapter7-vis_cfg/tutorial.md) | [官方文档: 渲染与配置](https://mujoco.readthedocs.io/en/latest/programming.html#visualization)
> > [vis_cfg(Python)](Python/Chapter6-vis_cfg/tutorial.md) | [官方文档: 渲染与配置](https://mujoco.readthedocs.io/en/latest/programming.html#visualization)
>
> 第八节   射线测距 
> - 测距传感器实现原理，自定义测距
> > [ray(C++)](CPP/Chapter8-ray/tutorial.md) | [官方文档: 射线检测](https://mujoco.readthedocs.io/en/latest/programming.html#ray-collisions)
> > [ray(Python)](Python/Chapter7-ray/tutorial.md) | [官方文档: 射线检测](https://mujoco.readthedocs.io/en/latest/programming.html#ray-collisions)
> 
### 拓展和进阶
> 触觉检测
> - 刚性触摸板和柔性材料触觉检测，通过api实现，并非插件方式
> > [touch(C++&Python)](extend/touch/readme.md) | [官方文档: 触觉传感器](https://mujoco.readthedocs.io/en/latest/XMLreference.html#sensor-touch)
>
> 软接触
> - mujoco中碰撞模型，如何通过调整动态“弹簧-阻尼”模型让模型展现不同材料的碰撞效果,对于碰撞和穿模问题如何调整
> > [soft contact(mjcf&C++&Python)](extend/soft_contact/tutorial.md) | [官方文档: 接触解算原理](https://mujoco.readthedocs.io/en/latest/computation/index.html#contacts)
>
> Ray Caster(基于ray)
> - 通过mujoco中的ray实现雷达、深度相机和自定义测距传感器
> > [ray caster](extend/deep_camera/readme.md) | [官方文档: 射线求交](https://mujoco.readthedocs.io/en/latest/programming.html#ray-collisions)
> > [独立仓库：Albusgive/mujoco_ray_caster](https://github.com/Albusgive/mujoco_ray_caster)
>
### 娱乐
> 红石模拟
> - 在 MuJoCo 中实现对 Minecraft（我的世界）红石电路及逻辑门的趣味三维仿真
> > [红石电路仿真(Python)](fun/mujoco_red_stone/README.md)

