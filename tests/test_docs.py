"""
Checks that every tagged function is documented in docs/code_reference/ (see CONTRIBUTING.md).

Run from the repo root:  python -m pytest tests/test_docs.py
"""
import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_FILES = [ROOT / "strategy_logic.py", *sorted((ROOT / "src").rglob("*.py"))]
DOCS_DIR = ROOT / "docs" / "code_reference"

ID = r"(?:ING|DAT|OPT|STR|EVT|BT|LIVE)-\d+(?:\.\d+)?[a-z]?"
CODE_TAG = re.compile(rf"^\s*\[({ID})\]")
DOC_HEADING = re.compile(rf"^### ({ID})\s+(\w+)", re.M)


def code_ids():
    """{id: (function name, file)} for every function whose docstring starts with a tag."""
    found, dupes = {}, []
    for path in CODE_FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                m = CODE_TAG.match(ast.get_docstring(node) or "")
                if m:
                    if m.group(1) in found:
                        dupes.append(m.group(1))
                    found[m.group(1)] = (node.name, path.relative_to(ROOT).as_posix())
    return found, dupes


def doc_entries():
    """{id: (function name, planned?)} for every '### ID name' heading in the docs."""
    entries = {}
    for path in sorted(DOCS_DIR.glob("*.md")):
        if path.name == "README.md":
            continue
        text = path.read_text(encoding="utf-8")
        headings = list(DOC_HEADING.finditer(text))
        for i, m in enumerate(headings):
            end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
            planned = "⬜ planned" in text[m.end():end]
            entries[m.group(1)] = (m.group(2), planned)
    return entries


def test_no_duplicate_ids():
    _, dupes = code_ids()
    assert not dupes, f"IDs used by more than one function: {dupes}"


def test_every_tagged_function_is_documented():
    code, _ = code_ids()
    docs = doc_entries()
    missing = [f"{i} {name} ({file})" for i, (name, file) in code.items() if i not in docs]
    assert not missing, "No docs entry in docs/code_reference/ for:\n  " + "\n  ".join(missing)


def test_doc_names_match_code():
    code, _ = code_ids()
    docs = doc_entries()
    wrong = [f"{i}: docs say {docs[i][0]}, code has {name}"
             for i, (name, _) in code.items() if i in docs and docs[i][0] != name]
    assert not wrong, "Docs heading names don't match the code:\n  " + "\n  ".join(wrong)


def test_no_stale_doc_entries():
    code, _ = code_ids()
    docs = doc_entries()
    stale = [f"{i} {name}" for i, (name, planned) in docs.items() if i not in code and not planned]
    assert not stale, ("Docs entries with no matching function in the code "
                       "(mark them '⬜ planned' or remove them):\n  " + "\n  ".join(stale))


if __name__ == "__main__":
    for t in [test_no_duplicate_ids, test_every_tagged_function_is_documented,
              test_doc_names_match_code, test_no_stale_doc_entries]:
        t()
    print("docs in sync")
