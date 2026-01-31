import ast
from pathlib import Path
from typing import Iterable

def iter_py_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*.py"):
        yield p

def get_defined_top_level_names(constants_path: Path) -> set[str]:
    tree = ast.parse(constants_path.read_text(encoding="utf-8"), filename=str(constants_path))
    defined: set[str] = set()

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    defined.add(t.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                defined.add(node.target.id)
        elif isinstance(node, ast.ClassDef):
            defined.add(node.name)
        elif isinstance(node, ast.FunctionDef):
            defined.add(node.name)
        elif isinstance(node, ast.AsyncFunctionDef):
            defined.add(node.name)

    return defined

def get_used_names_within_file(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    used: set[str] = set()

    class V(ast.NodeVisitor):
        def visit_Name(self, node: ast.Name):
            used.add(node.id)

        def visit_Attribute(self, node: ast.Attribute):
            # Attribute name itself (e.g., UpperBodyIdx.NOSE -> adds "NOSE")
            used.add(node.attr)
            self.generic_visit(node)

    V().visit(tree)
    return used

def get_outside_usage(src_root: Path, constants_module: str, constants_path: Path) -> set[str]:
    """
    Returns names referenced outside constants.py via:
      - from constants import NAME
      - import constants as C; C.NAME
      - import src.utils.landmarks.constants; src.utils.landmarks.constants.NAME
    """
    used_outside: set[str] = set()

    for py in iter_py_files(src_root):
        if py.resolve() == constants_path.resolve():
            continue

        try:
            tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        except SyntaxError:
            # skip broken files
            continue

        # Track aliases that refer to the constants module (e.g., C)
        module_aliases: set[str] = set()

        class V(ast.NodeVisitor):
            def visit_ImportFrom(self, node: ast.ImportFrom):
                if node.module == constants_module:
                    for alias in node.names:
                        # "from X import *" => cannot know; treat as "everything used"
                        if alias.name == "*":
                            used_outside.add("__STAR_IMPORT__")
                        else:
                            # record imported name (original, not local alias)
                            used_outside.add(alias.name)
                self.generic_visit(node)

            def visit_Import(self, node: ast.Import):
                for alias in node.names:
                    if alias.name == constants_module:
                        module_aliases.add(alias.asname or alias.name.split(".")[-1])
                self.generic_visit(node)

            def visit_Attribute(self, node: ast.Attribute):
                # Match C.SOMETHING when C is an alias for constants module
                if isinstance(node.value, ast.Name) and node.value.id in module_aliases:
                    used_outside.add(node.attr)
                self.generic_visit(node)

        V().visit(tree)

    # If star import exists anywhere, we can’t safely claim anything is unused.
    if "__STAR_IMPORT__" in used_outside:
        return {"__STAR_IMPORT__"}

    return used_outside

def find_repo_root(start: Path) -> Path:
    """
    Walk upward until we find a directory containing 'src/'.
    """
    p = start
    while True:
        if (p / "src").is_dir():
            return p
        if p.parent == p:
            raise RuntimeError("Could not find repo root containing a 'src/' folder.")
        p = p.parent

def main():
    # BEFORE: repo = Path(__file__).resolve().parents[1]
    repo = find_repo_root(Path(__file__).resolve())

    constants_path = repo / "src/utils/landmarks/constants.py"
    src_root = repo / "src"
    constants_module = "src.utils.landmarks.constants"

    defined = get_defined_top_level_names(constants_path)
    used_inside = get_used_names_within_file(constants_path)
    used_outside = get_outside_usage(src_root, constants_module, constants_path)

    if used_outside == {"__STAR_IMPORT__"}:
        print("Found 'from src.utils.landmarks.constants import *' somewhere.")
        print("Cannot reliably determine unused names until that is removed.")
        return

    used_outside = {n for n in used_outside if n in defined}

    used_outside_sorted = sorted(used_outside)
    only_inside = sorted((defined - used_outside) & used_inside)
    unused_even_inside = sorted((defined - used_outside) - used_inside)

    print("=== USED_OUTSIDE ===")
    for n in used_outside_sorted:
        print(n)

    print("\n=== ONLY_USED_INSIDE_CONSTANTS (candidate to inline/compact) ===")
    for n in only_inside:
        print(n)

    print("\n=== UNUSED_EVEN_INSIDE (candidate to delete) ===")
    for n in unused_even_inside:
        print(n)

if __name__ == "__main__":
    main()