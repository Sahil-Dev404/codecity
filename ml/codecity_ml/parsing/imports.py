"""Extract import/dependency relationships between files.

Produces raw import statements (import.py's job) then resolves them to
actual file paths within the repo (resolve_imports' job), becoming the
graph's import edges in graph/edges.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from codecity_ml.parsing.treesitter import parse_file


@dataclass(frozen=True)
class RawImport:
    """One import statement as written, before resolving to a real file."""

    source_file: str  # repo-relative path of the file containing the import
    module: str  # e.g. "os.path", "./utils", "../components/Button"
    is_relative: bool


@dataclass(frozen=True)
class ImportEdge:
    """A resolved import: source_file imports target_file, both repo-relative."""

    source_file: str
    target_file: str


def _extract_python_imports(root_node, source_file: str) -> list[RawImport]:
    imports: list[RawImport] = []

    def walk(node):
        if node.type == "import_statement":
            for child in node.named_children:
                if child.type == "dotted_name":
                    imports.append(
                        RawImport(source_file, child.text.decode(), is_relative=False)
                    )
        elif node.type == "import_from_statement":
            module_node = node.child_by_field_name("module_name")
            if module_node is not None:
                text = module_node.text.decode()
                imports.append(RawImport(source_file, text, is_relative=text.startswith(".")))
        for child in node.children:
            walk(child)

    walk(root_node)
    return imports


def _extract_js_ts_imports(root_node, source_file: str) -> list[RawImport]:
    imports: list[RawImport] = []

    def walk(node):
        # import X from "./foo"  |  export ... from "./foo"
        if node.type in ("import_statement", "export_statement"):
            for child in node.named_children:
                if child.type == "string":
                    text = child.text.decode().strip("'\"")
                    imports.append(
                        RawImport(source_file, text, is_relative=text.startswith("."))
                    )
        # const x = require("./foo")
        elif node.type == "call_expression":
            fn = node.child_by_field_name("function")
            if fn is not None and fn.text.decode() == "require":
                args = node.child_by_field_name("arguments")
                if args is not None and args.named_children:
                    text = args.named_children[0].text.decode().strip("'\"")
                    imports.append(
                        RawImport(source_file, text, is_relative=text.startswith("."))
                    )
        for child in node.children:
            walk(child)

    walk(root_node)
    return imports


def extract_imports(path: Path, repo_root: Path) -> list[RawImport]:
    """Extract raw import statements from one file. Returns [] for
    unsupported or unparseable files rather than raising."""
    parsed = parse_file(path)
    if parsed is None:
        return []

    root_node, lang = parsed
    source_file = path.relative_to(repo_root).as_posix()

    if lang == "python":
        return _extract_python_imports(root_node, source_file)
    if lang in ("javascript", "typescript", "tsx"):
        return _extract_js_ts_imports(root_node, source_file)
    return []


def _resolve_python_module(module: str, source_file: str, known_files: set[str]) -> str | None:
    """Try to resolve a dotted Python module path (e.g. 'app.services.foo')
    to a known repo file, trying both package (__init__.py) and module
    (foo.py) forms."""
    candidate = module.replace(".", "/")
    for suffix in ("/__init__.py", ".py"):
        path = candidate + suffix
        if path in known_files:
            return path
        # also try resolving relative to the source file's package
        pkg_relative = str(PurePosixPath(source_file).parent / path)
        if pkg_relative in known_files:
            return pkg_relative
    return None


def _resolve_relative_js(module: str, source_file: str, known_files: set[str]) -> str | None:
    """Resolve a relative JS/TS import ('./foo', '../bar') against the
    importing file's directory, trying common extensions and index files."""
    base = PurePosixPath(source_file).parent / module
    candidates = [
        f"{base}.ts", f"{base}.tsx", f"{base}.js", f"{base}.jsx",
        f"{base}/index.ts", f"{base}/index.tsx", f"{base}/index.js",
    ]
    # Normalize away any ".." segments
    for c in candidates:
        normalized = str(PurePosixPath(c))
        if normalized in known_files:
            return normalized
    return None


def resolve_imports(raw_imports: list[RawImport], known_files: set[str]) -> list[ImportEdge]:
    """Resolve RawImports to ImportEdges pointing at real files in the repo.

    Imports that can't be resolved (external packages like 'requests' or
    'react', or paths we fail to match) are silently dropped — we only want
    edges between files that actually exist in this repo's graph.
    """
    edges: list[ImportEdge] = []

    for raw in raw_imports:
        target: str | None = None

        if raw.is_relative and raw.module.startswith("."):
            # Could be Python ('.utils', '..models') or JS ('./utils')
            if raw.module.replace(".", "").startswith("/") or "/" in raw.module:
                target = _resolve_relative_js(raw.module, raw.source_file, known_files)
            else:
                target = _resolve_python_module(
                    raw.module.lstrip("."), raw.source_file, known_files
                )
        elif raw.is_relative:
            target = _resolve_relative_js(raw.module, raw.source_file, known_files)
        else:
            target = _resolve_python_module(raw.module, raw.source_file, known_files)

        if target is not None and target != raw.source_file:
            edges.append(ImportEdge(source_file=raw.source_file, target_file=target))

    return edges