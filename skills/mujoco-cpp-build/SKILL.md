---
name: mujoco-cpp-build
description: Use when building C++ examples, compiling simulation executables, configuring CMake, or resolving C++ building errors in the MuJoCo learning repository. It requires the agent to actively prompt the user for build requirements (source vs release path, target directories).
---

# MuJoCo C++ Build Skill

Use this skill to automate the building of C++ chapters and simulate tools in the repository. This supports both **Source-built Install** (e.g., `/opt/mujoco`) and **Pre-compiled Release** (custom directory) options.

## Response Style

- Keep answers concise and compact. If the active agent is framed as a chatbot, avoid excessive headings, blank lines, and long segmented explanations; prefer dense short paragraphs or a small bullet list.
- When mentioning a dependency, official API, external project, or useful reference, include a clickable link when one is known.
- When referencing this tutorial repository in a chat-only context, prefer GitHub links to project files instead of local absolute paths. Use repository-relative paths only when working inside a local clone.
- If the build/debug task involves ray caster examples, mention the maintained project when relevant: [Albusgive/mujoco_ray_caster](https://github.com/Albusgive/mujoco_ray_caster).

## CRITICAL: Active Prompting Requirement
Before starting any compilation or editing of `CMakeLists.txt`, **you must actively ask the user** for their compilation requirements. Use a clear, formatted message (or ask_question tool if appropriate) to align on:
1. **MuJoCo Library Choice**:
   - Option A: Source-built and installed to standard path `/opt/mujoco` (using `find_package(mujoco REQUIRED PATHS /opt/mujoco/lib/cmake NO_DEFAULT_PATH)`).
   - Option B: Pre-compiled Release or Source-built in a custom directory (e.g. `/path/to/mujoco-3.3.1`).
2. **Custom Directory Path** (Only if Option B is chosen): Ask the user to provide the absolute path to their MuJoCo directory.
3. **Target Chapter**: Which chapter or executable to compile (e.g., `CPP/Chapter1-make/basic`, `CPP/Chapter2-view&step`, or "All").

## Workflow

1. **Prompt the User**: Present the build options clearly and wait for their input.
2. **Locate the Workspace**: Resolve the repository path dynamically (using `scripts/find_repo.py`).
3. **Configure CMakeLists.txt**:
   - Modify the target chapter's `CMakeLists.txt` based on the user's choices.
   - For **Option A** (Installed under `/opt`):
     ```cmake
     set(MUJOCO_FOLDER /opt/mujoco/lib/cmake)
     find_package(mujoco REQUIRED PATHS ${MUJOCO_FOLDER} NO_DEFAULT_PATH)
     target_link_libraries(your_target mujoco::mujoco glut GL GLU glfw)
     ```
   - For **Option B** (Custom path, e.g., `/path/to/mujoco`):
     ```cmake
     set(MUJOCO_PATH "/path/to/mujoco")
     include_directories(${MUJOCO_PATH}/include)
     # For Source-built custom paths:
     link_directories(${MUJOCO_PATH}/build/bin)
     set(MUJOCO_LIB ${MUJOCO_PATH}/build/lib/libmujoco.so)
     # For Release pre-compiled paths:
     # link_directories(${MUJOCO_PATH}/bin)
     # set(MUJOCO_LIB ${MUJOCO_PATH}/lib/libmujoco.so)

     target_link_libraries(your_target ${MUJOCO_LIB} glut GL GLU glfw)
     ```
4. **Compile the Executable**:
   - Create a `build` directory under the target chapter (e.g. `CPP/Chapter1-make/build/`).
   - Run `cmake ..` and `make` (or `ninja`).
5. **Handle Dependency Errors**:
   - If missing `GLFW` or `GL` headers: inform the user and suggest running `sudo apt-get install libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev freeglut3-dev`.

## Validation

After compilation, verify by running the binary with a sample model if available:
```bash
./basic ../../../API-MJCF/pointer.xml
```
Verify stdout returns no errors and simulation runs.
