#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path


REQUIRED = ("directory.md", "MJCF", "Python", "CPP")
SKILL_ROOT = Path(__file__).resolve().parents[1]
LOCAL_CONFIG = SKILL_ROOT / ".local" / "repo_path.txt"


def is_repo_root(path: Path) -> bool:
    return all((path / item).exists() for item in REQUIRED)


def ancestors(path: Path):
    current = path.resolve()
    yield current
    yield from current.parents


def saved_repo_path() -> Path | None:
    if not LOCAL_CONFIG.exists():
        return None
    value = LOCAL_CONFIG.read_text(encoding="utf-8").strip()
    if not value:
        return None
    return Path(value).expanduser()


def candidates() -> list[Path]:
    items: list[Path] = []
    saved_root = saved_repo_path()
    if saved_root:
        items.append(saved_root)

    env_root = os.environ.get("MUJOCO_LEARNING_ROOT")
    if env_root:
        items.append(Path(env_root).expanduser())

    for base in ancestors(Path.cwd()):
        items.append(base)

    script_path = Path(__file__).resolve()
    items.extend(script_path.parents)

    home = Path.home()
    items.extend(
        [
            home / "mujoco_learning",
            home / "mujoco-learning",
            home / "code" / "mujoco_learning",
            home / "projects" / "mujoco_learning",
            home / "workspace" / "mujoco_learning",
        ]
    )
    return items


def save_repo_path(path: Path) -> int:
    root = path.expanduser().resolve()
    if not is_repo_root(root):
        print(f"Invalid MuJoCo tutorial repository path: {root}")
        print("Expected directory.md plus MJCF/, Python/, and CPP/ under that path.")
        return 1
    LOCAL_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    LOCAL_CONFIG.write_text(str(root), encoding="utf-8")
    print(root)
    return 0


def clear_repo_path() -> int:
    if LOCAL_CONFIG.exists():
        LOCAL_CONFIG.unlink()
    print("Cleared saved MuJoCo tutorial repository path.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Find or configure the local MuJoCo tutorial repository.")
    parser.add_argument("--set", dest="set_path", help="Save the local MuJoCo tutorial repository path.")
    parser.add_argument("--clear", action="store_true", help="Clear the saved repository path.")
    args = parser.parse_args()

    if args.clear:
        return clear_repo_path()
    if args.set_path:
        return save_repo_path(Path(args.set_path))

    for item in candidates():
        if is_repo_root(item):
            print(item.resolve())
            return 0

    if LOCAL_CONFIG.exists():
        print("Saved MuJoCo tutorial repository path is invalid or unavailable. Ask the user for the new clone path and run this script with --set PATH.")
        return 1

    print(
        "MuJoCo tutorial repository not found. Ask the user for the local clone path and run this script with --set PATH.",
        end="",
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
