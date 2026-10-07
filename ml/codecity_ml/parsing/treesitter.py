"""Shared tree-sitter setup: one parser per supported language.

Centralized so every module that needs an AST (imports.py today, possibly
function-level extraction later) gets a consistently configured parser
instead of each reinventing language setup.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import tree_sitter_javascript as tsjavascript
import tree_sitter_python as tspython
import tree_sitter_typescript as tstypescript
from tree_sitter import Language, Node, Parser

_SUFFIX_TO_LANG = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
}


@lru_cache(maxsize=None)
def _language(name: str) -> Language:
    """Load (and cache) the tree-sitter Language for `name`."""
    if name == "python":
        return Language(tspython.language())
    if name == "javascript":
        return Language(tsjavascript.language())
    if name == "typescript":
        return Language(tstypescript.language_typescript())
    if name == "tsx":
        return Language(tstypescript.language_tsx())
    raise ValueError(f"Unsupported language: {name}")


@lru_cache(maxsize=None)
def _parser(name: str) -> Parser:
    parser = Parser(_language(name))
    return parser


def language_for_suffix(suffix: str) -> str | None:
    """Map a file suffix (e.g. '.py') to a tree-sitter language name, or
    None if we don't support it."""
    return _SUFFIX_TO_LANG.get(suffix.lower())


def parse_file(path: Path) -> tuple[Node, str] | None:
    """Parse `path` and return (root_node, language_name), or None if the
    file's extension isn't supported or it can't be read as text."""
    lang_name = language_for_suffix(path.suffix)
    if lang_name is None:
        return None

    try:
        source = path.read_bytes()
    except OSError:
        return None

    tree = _parser(lang_name).parse(source)
    return tree.root_node, lang_name