# mujoco教程
MJCF为建模部分
CPP和python均为开发接口
extend为拓展和进阶

## 本地 Web 文档查看环境安装（可选）
如果只想在线浏览 GitHub 上的 Markdown，可以跳过本节。若希望在本机启动 Web 文档站查看教程，推荐使用 Python 3.10 或更新版本。项目已经提供 `pyproject.toml`，执行 `pip install -e .` 会以可编辑模式安装当前仓库，并自动安装 Python 版 MuJoCo 和本地文档站依赖。

<details open>
<summary><strong>pip / venv 安装</strong></summary>

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

</details>

<details>
<summary><strong>uv 安装</strong></summary>

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

</details>

<details>
<summary><strong>conda 安装</strong></summary>

```bash
conda create -n mujoco-learning python=3.10 -y
conda activate mujoco-learning
python -m pip install --upgrade pip
pip install -e .
```

如果遇到 `GLIBCXX_x.x.xx not found` 之类的问题，可以在 conda 环境中尝试：

```bash
conda install -c conda-forge libstdcxx-ng
```

</details>

安装完成后可以验证 MuJoCo viewer：

```bash
python -m mujoco.viewer
```

也可以指定一个 MJCF 文件启动：

```bash
python -m mujoco.viewer --mjcf=API-MJCF/mecanum.xml
```

更多 MuJoCo 安装说明见 [mujoco安装](MJCF/Chapter0-install/tutorial.md)。

## 本地文档站
安装依赖后，在仓库根目录启动 FastAPI 文档站：

```bash
mujoco_learning_doc
```

默认会从 `127.0.0.1:8000` 启动；如果 8000 已被占用，会自动尝试下一个可用端口。也可以手动指定端口：

```bash
mujoco_learning_doc --port 8001
```

或者使用环境变量：

```bash
MUJOCO_DOCS_PORT=8001 mujoco_learning_doc
```

也可以直接使用 uvicorn：

```bash
uvicorn mujoco_learning_doc.main:app --reload --host 127.0.0.1 --port 8000
```

浏览器打开 <http://127.0.0.1:8000> 即可查看本仓库的 Markdown 教程。文档站会自动读取 `README.md`、`directory.md` 以及各章节的 `tutorial.md` / `readme.md`，并支持目录导航、搜索、图片显示和代码高亮。

## Codex Skills
本仓库在 `skills/` 下提供两个 Codex skill：

- `skills/mujoco-teaching`：教学型。用于向 agent 提问 MuJoCo 概念、教程内容、学习路线和 API 原理。
- `skills/mujoco-engineering`：工程型。用于让 agent 复现示例、开发 MJCF/Python/C++ MuJoCo 功能、查找对应教程代码路径。

发布到 GitHub 后，可用 Codex 的 `skill-installer` 从仓库路径安装：

```bash
python scripts/install-skill-from-github.py --repo Albusgive/mujoco_learning --path skills/mujoco-teaching skills/mujoco-engineering
```

安装完成后重启 Codex。skill 内只使用仓库相对路径，不包含开发电脑的绝对路径。

如果 Codex 不在本仓库工作区内运行，skill 会先尝试自动寻找本仓库。首次找不到时，agent 会询问本机仓库路径，并保存到已安装 skill 目录下的本地配置文件：

```text
<installed-skill>/.local/repo_path.txt
```

`.local/repo_path.txt` 是普通文件路径，Python 脚本会用当前系统的路径规则读写，Windows、macOS、Linux 都可用。不同系统可按下面方式手动设置：

<details>
<summary><strong>macOS / Linux</strong></summary>

```bash
python ~/.codex/skills/mujoco-engineering/scripts/find_repo.py --set /path/to/mujoco_learning
python ~/.codex/skills/mujoco-teaching/scripts/find_repo.py --set /path/to/mujoco_learning
```

清除保存路径：

```bash
python ~/.codex/skills/mujoco-engineering/scripts/find_repo.py --clear
python ~/.codex/skills/mujoco-teaching/scripts/find_repo.py --clear
```

</details>

<details>
<summary><strong>Windows PowerShell</strong></summary>

```powershell
python $HOME\.codex\skills\mujoco-engineering\scripts\find_repo.py --set C:\path\to\mujoco_learning
python $HOME\.codex\skills\mujoco-teaching\scripts\find_repo.py --set C:\path\to\mujoco_learning
```

清除保存路径：

```powershell
python $HOME\.codex\skills\mujoco-engineering\scripts\find_repo.py --clear
python $HOME\.codex\skills\mujoco-teaching\scripts\find_repo.py --clear
```

</details>

<details>
<summary><strong>Windows CMD</strong></summary>

```bat
python %USERPROFILE%\.codex\skills\mujoco-engineering\scripts\find_repo.py --set C:\path\to\mujoco_learning
python %USERPROFILE%\.codex\skills\mujoco-teaching\scripts\find_repo.py --set C:\path\to\mujoco_learning
```

清除保存路径：

```bat
python %USERPROFILE%\.codex\skills\mujoco-engineering\scripts\find_repo.py --clear
python %USERPROFILE%\.codex\skills\mujoco-teaching\scripts\find_repo.py --clear
```

</details>

如果保存的路径失效，agent 会重新询问并覆盖保存。

环境变量 `MUJOCO_LEARNING_ROOT` 仍可作为可选备用方案。不同系统设置方式如下：

<details>
<summary><strong>Windows PowerShell</strong></summary>

当前会话临时设置：

```powershell
$env:MUJOCO_LEARNING_ROOT = "C:\path\to\mujoco_learning"
```

永久设置：

```powershell
[Environment]::SetEnvironmentVariable("MUJOCO_LEARNING_ROOT", "C:\path\to\mujoco_learning", "User")
```

</details>

<details>
<summary><strong>Windows CMD</strong></summary>

当前会话临时设置：

```bat
set MUJOCO_LEARNING_ROOT=C:\path\to\mujoco_learning
```

永久设置：

```bat
setx MUJOCO_LEARNING_ROOT "C:\path\to\mujoco_learning"
```

</details>

<details>
<summary><strong>macOS / Linux</strong></summary>

```bash
export MUJOCO_LEARNING_ROOT=/path/to/mujoco_learning
```

</details>

两个 skill 也会自动尝试从当前工作区和常见克隆目录中寻找本仓库。

## 教程目录
🚀[教程目录（更新中）](directory.md)
## 视频教程
📺[视频教程链接](https://www.bilibili.com/video/BV1wMdHYVEnx/?spm_id_from=333.1387.collection.video_card.click&vd_source=71e0e4952bb37bdc39eaabd9c08be754)
## 教程封面
**建模部分封面**
![MJCF](MJCF/asset/封面.png)
**Python API部分封面**
![py](MJCF/asset/封面-py.png)
**C++ API部分封面**
![cpp](MJCF/asset/封面-cpp.png)
## 技术交流
![](MJCF/asset/mujoco交流群.jpg)
