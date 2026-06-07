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

## Agent Skills (智能体技能安装)
本仓库在 `skills/` 下提供三个智能体技能（Skills）：

- `skills/mujoco-teaching`：教学型。用于向 Agent 提问 MuJoCo 概念、教程内容、学习路线和 API 原理。
- `skills/mujoco-engineering`：工程型。用于让 Agent 复现示例、开发 MJCF/Python/C++ MuJoCo 功能、查找对应教程代码路径。
- `skills/mujoco-cpp-build`：构建型。一键编译构建本仓库 C++ 实例与工具，支持源码编译版与 Release 依赖版，并在编译前主动向用户提问构建需求。

为了让不同平台的智能体（Agent）能识别和使用这些技能，我们提供了以下几种安装配置方式：

### 1. Antigravity (Gemini Advanced Agentic Coding)
* **工作区自动加载**：当您在当前仓库根目录下运行 Antigravity 时，CLI 将自动扫描并加载 `skills/` 目录下的所有技能，无需额外安装。
* **全局安装**：如果您在其他工作区也想让 Antigravity 调用这些技能，可将其复制到系统全局插件目录：
  ```bash
  mkdir -p ~/.gemini/config/skills
  cp -r skills/mujoco-teaching skills/mujoco-engineering skills/mujoco-cpp-build ~/.gemini/config/skills/
  ```

### 2. Claude Code
* **工作区级适配**：Claude Code 会自动加载项目根目录下的 `.claude/skills/` 适配器。这些适配器已经配置好并指向了 `skills/` 下的 canonical 技能文件。
* **全局安装**：如果要在其他目录下也能够使用，可将 canonical 目录直接安装至 Claude 的全局路径下：
  ```bash
  mkdir -p ~/.claude/skills
  cp -r skills/mujoco-teaching skills/mujoco-engineering skills/mujoco-cpp-build ~/.claude/skills/
  ```

### 3. OpenCode
* **工作区级适配**：OpenCode 会自动加载项目根目录下的 `.opencode/skills/` 适配器。
* **全局安装**：如果需要全局使用，复制技能文件夹至 OpenCode 全局技能目录：
  ```bash
  mkdir -p ~/.opencode/skills
  cp -r skills/mujoco-teaching skills/mujoco-engineering skills/mujoco-cpp-build ~/.opencode/skills/
  ```

### 4. Codex
* **通过 skill-installer 从 GitHub 远程安装**：
  使用 Codex 的 `skill-installer` 直接从 GitHub 地址进行安装：
  ```bash
  skill-installer install https://github.com/Albusgive/mujoco_learning/tree/main/skills/mujoco-teaching
  skill-installer install https://github.com/Albusgive/mujoco_learning/tree/main/skills/mujoco-engineering
  skill-installer install https://github.com/Albusgive/mujoco_learning/tree/main/skills/mujoco-cpp-build
  ```
* **本地手动安装**：直接将技能文件夹拷贝至 Codex 本地技能存储路径：
  ```bash
  mkdir -p ~/.codex/skills
  cp -r skills/mujoco-teaching skills/mujoco-engineering skills/mujoco-cpp-build ~/.codex/skills/
  ```

安装/复制完成后，请重启对应的命令行工具或 Agent 客户端。

---

## 跨工作区运行时的仓库路径设置
如果智能体（如 Antigravity / Codex / Claude Code 等）未在当前仓库的工作区中运行，技能会在加载时尝试动态搜寻本仓库。首次找不到时，智能体主动询问您本仓库的克隆路径，并自动将其保存到已安装技能目录下的本地配置文件中：

```text
<installed-skill>/.local/repo_path.txt
```

`.local/repo_path.txt` 是普通路径文本文件。如果需要，您也可以通过运行技能中的 `find_repo.py` 脚本来手动设置或清除该缓存路径。不同系统设置方式如下：

<details>
<summary><strong>macOS / Linux</strong></summary>

```bash
python ~/.codex/skills/mujoco-engineering/scripts/find_repo.py --set /path/to/mujoco_learning
python ~/.codex/skills/mujoco-teaching/scripts/find_repo.py --set /path/to/mujoco_learning
python ~/.codex/skills/mujoco-cpp-build/scripts/find_repo.py --set /path/to/mujoco_learning
```

清除保存路径：

```bash
python ~/.codex/skills/mujoco-engineering/scripts/find_repo.py --clear
python ~/.codex/skills/mujoco-teaching/scripts/find_repo.py --clear
python ~/.codex/skills/mujoco-cpp-build/scripts/find_repo.py --clear
```

</details>

<details>
<summary><strong>Windows PowerShell</strong></summary>

```powershell
python $HOME\.codex\skills\mujoco-engineering\scripts\find_repo.py --set C:\path\to\mujoco_learning
python $HOME\.codex\skills\mujoco-teaching\scripts\find_repo.py --set C:\path\to\mujoco_learning
python $HOME\.codex\skills\mujoco-cpp-build\scripts\find_repo.py --set C:\path\to\mujoco_learning
```

清除保存路径：

```powershell
python $HOME\.codex\skills\mujoco-engineering\scripts\find_repo.py --clear
python $HOME\.codex\skills\mujoco-teaching\scripts\find_repo.py --clear
python $HOME\.codex\skills\mujoco-cpp-build\scripts\find_repo.py --clear
```

</details>

<details>
<summary><strong>Windows CMD</strong></summary>

```bat
python %USERPROFILE%\.codex\skills\mujoco-engineering\scripts\find_repo.py --set C:\path\to\mujoco_learning
python %USERPROFILE%\.codex\skills\mujoco-teaching\scripts\find_repo.py --set C:\path\to\mujoco_learning
python %USERPROFILE%\.codex\skills\mujoco-cpp-build\scripts\find_repo.py --set C:\path\to\mujoco_learning
```

清除保存路径：

```bat
python %USERPROFILE%\.codex\skills\mujoco-engineering\scripts\find_repo.py --clear
python %USERPROFILE%\.codex\skills\mujoco-teaching\scripts\find_repo.py --clear
python %USERPROFILE%\.codex\skills\mujoco-cpp-build\scripts\find_repo.py --clear
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
