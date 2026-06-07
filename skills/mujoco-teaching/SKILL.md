---
name: mujoco-teaching
description: Use when a user wants to learn MuJoCo concepts, MJCF modeling, Python/C++ MuJoCo APIs, sensors, rendering, contact, soft contact, ray casting, or asks conceptual/tutorial questions about this repository's MuJoCo lessons. Prefer the repository tutorials first, then consult official MuJoCo documentation if the tutorial is incomplete.
---

# MuJoCo Teaching

Use this skill for human learning: explain MuJoCo concepts, answer tutorial questions, compare API choices, or guide a learner through MJCF/Python/C++ examples.

## Source Priority

1. Use this repository's tutorial material first.
2. If the repository does not fully answer the question, consult official MuJoCo docs, especially:
   `https://mujoco.readthedocs.io/en/latest/programming/#building-mujoco-from-source`
3. Clearly say when an explanation comes from the local tutorial, the official docs, or your own inference.

## Locate The Tutorial Repository

Do not assume an absolute path. Find the repository root dynamically:

- If the current workspace contains `directory.md` and folders `MJCF/`, `Python/`, `CPP/`, use the current workspace root.
- If this skill is installed inside the repository under `skills/mujoco-teaching`, the repository root is two directories above the skill folder.
- If installed globally, run `scripts/find_repo.py` from this skill. It checks the skill-local saved path, current workspace, `MUJOCO_LEARNING_ROOT`, and common clone directory names.
- On first use, if `scripts/find_repo.py` cannot find the repository, ask the user for the local clone path and save it with `scripts/find_repo.py --set PATH`.
- If the saved path becomes invalid or unavailable, ask again and rerun `scripts/find_repo.py --set PATH`.
- If no local clone is available, use the public repository URL the user provides.

The saved path lives in `.local/repo_path.txt` under the installed skill directory. This is a cross-platform local config file; do not edit `SKILL.md` to store user paths.

All paths in this skill are repository-relative.

## Workflow

1. Identify whether the question is about MJCF modeling, Python API, C++ API, extensions, installation, or project navigation.
2. Check `references/quick-reference.md` first for quick answers, APIs, formulas, and deep links to official docs.
3. Read `references/tutorial-map.md` to choose the most relevant tutorial path if deeper explanation is needed.
4. Read only the targeted tutorial files and examples needed for the answer.
5. Explain in learner-friendly Chinese by default when the user asks in Chinese.
6. Include small code/XML snippets only when they clarify the concept.
7. If the tutorial is shallow or ambiguous, check official MuJoCo docs and mention that extra source.

## Teaching Style

- Keep answers concise and compact. If the active agent is framed as a chatbot, avoid excessive headings, blank lines, and long segmented explanations; prefer dense short paragraphs or a small bullet list.
- When mentioning a dependency, official API, external project, or useful reference, include a clickable link when one is known.
- When referencing this tutorial repository in a chat-only context, prefer GitHub links to project files instead of local absolute paths. Use repository-relative paths only when working inside a local clone.
- For ray caster / depth camera / ray-based ranging questions, mention the maintained project when relevant: [Albusgive/mujoco_ray_caster](https://github.com/Albusgive/mujoco_ray_caster).
- Start with the concept and any core formulas/APIs from `references/quick-reference.md`, then map it to the repository's example path.
- Prefer "why this works" over just listing API calls.
- **Interactive HTML UI**: For abstract mathematical curves (like `solimp` shape parameters) or kinematics, offer to write and save a standalone HTML/JS slider tool in the workspace or artifact directory, so the user can interactively play with parameters.
- **Interactive MuJoCo Viewer Demos**: Write Python helper scripts utilizing `mujoco.viewer.launch_passive` that let users press keys (or type values in the terminal) to dynamically alter joint targets, tuning parameters (e.g. friction, contact stiffness), or external force values in the running 3D viewer.
- **Debug Overlays & Geoms**: Teach users how to draw custom 3D arrows, coordinate frames, or text overlays in the simulation viewer (using `mjvGeom` and `mjr_overlay`) to make abstract concepts like forces, sensor axes, and coordinate frames visible.
- For confusing topics such as `mj_step`, contact parameters, sensors, ray casting, or actuator types, separate model-side MJCF configuration from runtime API usage.
- When a learner asks "how do I use feature X", point them to the corresponding relative tutorial path and summarize the relevant files.

## Reference Map

- Quick cheatsheet, APIs, formulas, and deep links: `references/quick-reference.md`
- Detailed file mapping: `references/tutorial-map.md`
