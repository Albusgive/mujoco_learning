from __future__ import annotations

import argparse
import os
import re
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import quote, unquote, urlsplit, urlunsplit

import markdown
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
STATIC_DIR = APP_DIR / "static"
TEMPLATES_DIR = APP_DIR / "templates"

DOC_FILENAMES = {"README.md", "readme.md", "tutorial.md", "directory.md"}
ROOT_DOCS = ("README.md", "directory.md")
NATURAL_SORT_RE = re.compile(r"(\d+)")
MARKDOWN_EXTENSIONS = [
    "extra",
    "toc",
    "tables",
    "fenced_code",
    "codehilite",
    "pymdownx.arithmatex",
    "sane_lists",
]
LINK_RE = re.compile(r"(!?\[[^\]]*?\]\()([^)]+)(\))")
HTML_LINK_RE = re.compile(r"""((?:src|href)=["'])([^"']+)(["'])""", re.IGNORECASE)


@dataclass(frozen=True)
class DocItem:
    title: str
    path: str
    section: str
    url: str


@dataclass(frozen=True)
class NavGroup:
    key: str
    title: str
    items: list[DocItem]
    is_open: bool


app = FastAPI(title="MuJoCo 教程文档站")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


def port_is_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def find_available_port(host: str, start_port: int, attempts: int = 20) -> int:
    for port in range(start_port, start_port + attempts):
        try:
            if port_is_available(host, port):
                return port
        except PermissionError:
            return start_port
    raise RuntimeError(f"No available port found from {start_port} to {start_port + attempts - 1}.")


def run() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Start the local MuJoCo tutorial docs site.")
    parser.add_argument("--host", default=os.environ.get("MUJOCO_DOCS_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("MUJOCO_DOCS_PORT", "8000")),
        help="Preferred port. If it is busy, the next free port will be used.",
    )
    parser.add_argument("--no-reload", action="store_true", help="Disable uvicorn reload mode.")
    args = parser.parse_args()

    port = find_available_port(args.host, args.port)
    if port != args.port:
        print(f"Port {args.port} is in use, using {port} instead.")
    print(f"Docs site: http://{args.host}:{port}")
    uvicorn.run("mujoco_learning_doc.main:app", host=args.host, port=port, reload=not args.no_reload)


def safe_repo_path(relative_path: str) -> Path:
    decoded = unquote(relative_path).strip("/")
    candidate = (REPO_ROOT / decoded).resolve()
    try:
        candidate.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="File not found") from exc
    return candidate


def web_path(path: str) -> str:
    return quote(path.replace("\\", "/"), safe="/")


def title_from_markdown(path: Path) -> str:
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                return stripped.lstrip("#").strip() or path.parent.name
    except UnicodeDecodeError:
        return path.stem
    return path.parent.name if path.name.lower() in {"tutorial.md", "readme.md"} else path.stem


def section_for(path: Path) -> str:
    relative = path.relative_to(REPO_ROOT)
    if len(relative.parts) == 1:
        return "首页"
    return relative.parts[0]


def section_title(section: str) -> str:
    titles = {
        "MJCF": "MJCF建模",
        "Python": "Python",
        "CPP": "CPP",
        "extend": "拓展和进阶",
        "fun": "娱乐",
        "API-MJCF": "API-MJCF",
        "utils": "工具",
        "首页": "首页",
    }
    return titles.get(section, section)


def section_order(section: str) -> tuple[int, str]:
    order = {
        "MJCF": 0,
        "Python": 1,
        "CPP": 2,
        "extend": 3,
        "fun": 4,
        "API-MJCF": 5,
        "utils": 6,
        "首页": 98,
    }
    return order.get(section, 50), section.lower()


def iter_markdown_docs() -> Iterable[Path]:
    for root_doc in ROOT_DOCS:
        path = REPO_ROOT / root_doc
        if path.exists():
            yield path

    for path in sorted(REPO_ROOT.rglob("*.md"), key=natural_sort_key):
        if any(part.startswith(".") for part in path.relative_to(REPO_ROOT).parts):
            continue
        if path.relative_to(REPO_ROOT).as_posix() in ROOT_DOCS:
            continue
        if path.name in DOC_FILENAMES or path.name.lower() in DOC_FILENAMES:
            yield path


def natural_sort_key(path: Path) -> list[str | int]:
    relative = path.relative_to(REPO_ROOT).as_posix().lower()
    return [int(part) if part.isdigit() else part for part in NATURAL_SORT_RE.split(relative)]


def build_docs() -> list[DocItem]:
    docs: list[DocItem] = []
    seen: set[str] = set()
    for path in iter_markdown_docs():
        relative = path.relative_to(REPO_ROOT).as_posix()
        if relative in seen:
            continue
        seen.add(relative)
        docs.append(
            DocItem(
                title=title_from_markdown(path),
                path=relative,
                section=section_for(path),
                url=f"/docs/{web_path(relative)}",
            )
        )
    return docs


def grouped_docs(docs: list[DocItem], active_path: str) -> list[NavGroup]:
    groups: dict[str, list[DocItem]] = {}
    for doc in docs:
        groups.setdefault(doc.section, []).append(doc)

    nav_groups: list[NavGroup] = []
    for section in sorted(groups, key=section_order):
        if section == "首页":
            continue
        items = groups[section]
        nav_groups.append(
            NavGroup(
                key=section,
                title=section_title(section),
                items=items,
                is_open=any(item.path == active_path for item in items),
            )
        )
    return nav_groups


def home_docs(docs: list[DocItem]) -> list[DocItem]:
    return [doc for doc in docs if doc.section == "首页"]


def is_external_url(url: str) -> bool:
    parsed = urlsplit(url)
    return bool(parsed.scheme or parsed.netloc) or url.startswith(("#", "mailto:", "tel:", "data:"))


def resolve_target(base_doc: Path, raw_target: str) -> str:
    target = raw_target.strip()
    if is_external_url(target):
        return target

    parsed = urlsplit(target)
    if not parsed.path:
        return target

    resolved = (base_doc.parent / unquote(parsed.path)).resolve()
    try:
        relative = resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return target

    suffix = resolved.suffix.lower()
    if suffix == ".md":
        path = f"/docs/{web_path(relative)}"
    else:
        path = f"/files/{web_path(relative)}"

    return urlunsplit(("", "", path, parsed.query, parsed.fragment))


def rewrite_markdown_links(markdown_text: str, base_doc: Path) -> str:
    def replace_markdown_link(match: re.Match[str]) -> str:
        prefix, target, suffix = match.groups()
        return f"{prefix}{resolve_target(base_doc, target)}{suffix}"

    return LINK_RE.sub(replace_markdown_link, markdown_text)


def rewrite_html_links(html: str, base_doc: Path) -> str:
    def replace_html_link(match: re.Match[str]) -> str:
        prefix, target, suffix = match.groups()
        return f"{prefix}{resolve_target(base_doc, target)}{suffix}"

    return HTML_LINK_RE.sub(replace_html_link, html)


def render_markdown(path: Path) -> str:
    source = path.read_text(encoding="utf-8")
    source = rewrite_markdown_links(source, path)
    html = markdown.markdown(
        source,
        extensions=MARKDOWN_EXTENSIONS,
        extension_configs={
            "codehilite": {"guess_lang": False},
            "pymdownx.arithmatex": {"generic": True},
        },
        output_format="html5",
    )
    return rewrite_html_links(html, path)


def render_doc(request: Request, relative_path: str) -> HTMLResponse:
    path = safe_repo_path(relative_path)
    if not path.exists() or not path.is_file() or path.suffix.lower() != ".md":
        raise HTTPException(status_code=404, detail="Document not found")

    docs = build_docs()
    active_path = path.relative_to(REPO_ROOT).as_posix()
    return templates.TemplateResponse(
        request,
        "doc.html",
        context={
            "request": request,
            "title": title_from_markdown(path),
            "content": render_markdown(path),
            "docs": docs,
            "home_docs": home_docs(docs),
            "groups": grouped_docs(docs, active_path),
            "active_path": active_path,
        },
    )


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return render_doc(request, "README.md")


@app.get("/docs/{relative_path:path}", response_class=HTMLResponse)
async def docs(request: Request, relative_path: str) -> HTMLResponse:
    return render_doc(request, relative_path)


@app.get("/files/{relative_path:path}")
async def files(relative_path: str) -> FileResponse:
    path = safe_repo_path(relative_path)
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path)
