# 3D绘制
&emsp;&emsp;mujoco提供显示基础几何体和mujoco提供的一些特殊渲染几何体。查看文档可知mjv_initGeom函数能在渲染场景中增加几何体，mjv_connector可以用mjv_initGeom初始化的几何体绘制提供的一些特殊形状（如箭头，直线等）。mujoco显示画面的原理是通过mjv_updateScene 将仿真数据储存到mjvScene中，这是已经处理好的几何数据，接下来使用mjr_render传递给opengl渲染。我们在绘制过程中是要在仿真的几何数据处理完之后，加入绘制信息，再交给opengl渲染。
&emsp;&emsp;在mjvScene中添加信息，其实是直接在mjvScene的geoms后面续写，而且要增加ngeom长度。这里通过注释可以理解，mjvScene根据ngeom确定几何体数量再从geoms中获取资源。

```c
struct _mjvScene {
    // abstract geoms
    int maxgeom;              // size of allocated geom buffer
    int ngeom;                // number of geoms currently in buffer
    mjvGeom* geoms;           // buffer for geoms
    int* geomorder;           // buffer for ordering geoms by distance to camera
    // ...
};
typedef struct _mjvScene mjvScene;
```

mjv_initGeom函数原型：

```c
void mjv_initGeom(mjvGeom* geom, int type, const mjtNum size[3],
                  const mjtNum pos[3], const mjtNum mat[9], const float rgba[4]);
```

mjv_connector函数原型：

```c
void mjv_connector(mjvGeom* geom, int type, mjtNum width,
                   const mjtNum from[3], const mjtNum to[3]);
```

geom是传入的仅绘制的几何体，需要使用mjv_initGeom初始化，type见下面，width是绘制的宽度，这个是对于渲染出来的画面的宽度，from起点，to终点

```c
typedef enum _mjtGeom {
    // ...
    mjGEOM_ARROW = 100,     // 箭头
    mjGEOM_ARROW1,          // 无楔形箭头
    mjGEOM_ARROW2,          // 双向箭头
    mjGEOM_LINE,            // 直线
    // ...
} mjtGeom;
```
这里是可以绘制的几何形状类型，分别是箭头，无楔形箭头，双向箭头，直线。
<font color=Green>*演示——绘制几何体函数：*</font>

```C++
void draw_geom(mjvScene *scn, int type,mjtNum *size, mjtNum *pos,mjtNum* mat, float rgba[4]) {
  scn->ngeom += 1;
  mjvGeom *geom = scn->geoms + scn->ngeom - 1;
  mjv_initGeom(geom, type, size, pos, mat, rgba);
}
......
mjtNum size[3] = {0.3, 0, 0};
mjtNum pos[3] = {0, 0, 1.0};
mjtNum mat[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
draw_geom(&scn, mjGEOM_SPHERE,size, pos, mat, color);
```

<font color=Green>*演示——绘制直线函数：*</font>

```C++
void draw_line(mjvScene *scn, mjtNum *from, mjtNum *to, mjtNum width,
               float* rgba) {
  scn->ngeom += 1;
  mjvGeom *geom = scn->geoms + scn->ngeom - 1;
  mjv_initGeom(geom, mjGEOM_SPHERE, NULL, NULL, NULL, rgba);
  mjv_connector(geom, mjGEOM_LINE, width, from, to);
}
......
mjtNum from[3] = {0, 0, 0};
mjtNum to[3] = {0, 1, 1};
float color[4] = {0, 1, 0, 1};
draw_line(&scn, from, to, 20, color);
```

<font color=Green>*演示——绘制箭头函数：*</font>

```C++
void draw_arrow(mjvScene *scn, mjtNum *from, mjtNum *to, mjtNum width,
                float rgba[4]) {
  scn->ngeom += 1;
  mjvGeom *geom = scn->geoms + scn->ngeom - 1;
  mjv_initGeom(geom, mjGEOM_SPHERE, NULL, NULL, NULL, rgba);
  mjv_connector(geom, mjGEOM_ARROW, width, from, to);
}
......
mjtNum from[3] = {0, 0, 0};
mjtNum to[3] = {0, 1, 1};
float color[4] = {0, 1, 0, 1};
draw_arrow(&scn, from, to, 20, color);
```

<font color=Green>*使用：*</font>

```C++
mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
draw_geom(&scn, mjGEOM_SPHERE,size, pos, mat, color);
draw_line(&scn, from, to, color);
draw_arrow(&scn, from, to, 20, color);
mjr_render(viewport, &scn, &con);
```
这里要注意在 mjv_updateScene函数之后，mjr_render函数之前调用。

# 2D绘制
字体尺寸的初始化：

```c
typedef enum _mjtFontScale {
    mjFONTSCALE_50  = 50,   // 50% 缩放，适用于低分辨率渲染
    mjFONTSCALE_100 = 100,  // 100% 缩放，默认普通比例
    mjFONTSCALE_150 = 150,  // 150% 缩放
    mjFONTSCALE_200 = 200   // 200% 缩放
} mjtFontScale;
```

查阅文档我们可知，2D 绘制函数需要在 `mjr_render` 之后进行调用：

> [!IMPORTANT]
> `mjr_render` 会在开始渲染时清除视口（Viewport）。因此，所有自定义的 2D 绘制操作（如绘制文本、矩形、表格或标签等）必须在调用 `mjr_render` **之后** 进行，否则绘制的内容会被视口清除覆盖。

<font color=Green>*绘制文本：*</font>

```C++
MJAPI void mjr_text(int font, const char* txt, const mjrContext* con,
                    float x, float y, float r, float g, float b);
```
font:字号，使用mjtFont中定义的    
txt：文本   
con：mjrContext   
x,y:渲染界面比例位置，取值[0-1)   
r,g,b:字体颜色    

<font color=Green>*绘制对应表格（overlay）：*</font>

```C++
MJAPI void mjr_overlay(int font, int gridpos, mjrRect viewport,
                       const char* overlay, const char* overlay2, const mjrContext* con);
```
font:字号，使用mjtFont中定义的    
gridpos：绘制位置，使用mjtGridPos中定义的     
mjrRect：mjrRect，界面矩形  
overlay：第一列     
overlay2：第二列      
con：mjrContext         

<font color=Green>*绘制矩形：*</font>

```C++
MJAPI void mjr_rectangle(mjrRect viewport, float r, float g, float b, float a);
```
mjrRect：mjrRect，矩形      
rgba:颜色   

<font color=Green>*绘制标签：*</font>

```C++
MJAPI void mjr_label(mjrRect viewport, int font, const char* txt,
                     float r, float g, float b, float a, float rt, float gt, float bt,
                     const mjrContext* con);
```
viewport：标签位置    
font:字号，使用mjtFont中定义的    
txt：文本   
rgba:标签底色   
rt,gt,bt:文字颜色   
con：mjrContext   
