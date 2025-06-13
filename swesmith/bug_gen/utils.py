import ast
import os
import subprocess

from dataclasses import dataclass
from typing import Any
from dotenv import load_dotenv
from itertools import combinations
from swesmith.constants import TEMP_PATCH
from swesmith.utils import generate_hash
import tempfile


load_dotenv()


DEVNULL = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
ENTITY_TYPES = {
    "class": [ast.ClassDef],
    "func": [ast.FunctionDef],
    "object": [ast.ClassDef, ast.FunctionDef],
}


class BugRewrite:
    cost: float = 0
    explanation: str = ""
    output: str
    rewrite: str
    strategy: str

    def __init__(
        self,
        rewrite: str,
        explanation: str,
        strategy: str,
        cost: float = 0,
        output: str = "",
    ):
        self.rewrite = rewrite
        self.explanation = explanation
        self.cost = cost
        self.strategy = strategy
        self.output = output

    def get_hash(self) -> str:
        """Generates a hash for the bug rewrite."""
        return generate_hash(self.rewrite)

    def to_dict(self) -> dict[str, Any]:
        """Converts the bug rewrite to a dictionary."""
        return {
            "cost": self.cost,
            "explanation": self.explanation,
            "output": self.output,
            "rewrite": self.rewrite,
            "strategy": self.strategy,
        }


@dataclass
class CodeEntity:
    """Data class to hold information about a code entity (e.g. function, class)."""

    file_path: str
    indent_level: int
    indent_size: int
    line_end: int
    line_start: int
    src_code: Any
    src_node: Any


def apply_code_change(candidate: CodeEntity, bug: BugRewrite) -> None:
    """Replaces lines in a file between start_line and end_line with replacement_code."""
    with open(candidate.file_path, "r") as file:
        lines = file.readlines()
    if (
        candidate.line_start < 1
        or candidate.line_end > len(lines)
        or candidate.line_start > candidate.line_end
    ):
        raise ValueError("Invalid line range specified.")
    replacement_lines = [
        f"{' ' * candidate.indent_level * candidate.indent_size}{x}"
        if len(x.strip()) > 0
        else x
        for x in bug.rewrite.splitlines(keepends=True)
    ]
    lines = (
        lines[: candidate.line_start - 1]
        + replacement_lines
        + lines[candidate.line_end :]
    )
    with open(candidate.file_path, "w") as file:
        file.writelines(lines)


def apply_patches(repo: str, patch_files: list[str]) -> str | None:
    """Apply multiple patches to a target local directory, and get the combined patch."""
    cwd = os.getcwd()
    os.chdir(repo)
    try:
        for patch_file in patch_files:
            subprocess.run(
                ["git", "apply", os.path.join("..", patch_file)], check=True, **DEVNULL
            )
        patch = get_patch(os.getcwd(), reset_changes=True)

        # Sanity check that merged patch applies cleanly
        with open(TEMP_PATCH, "w") as f:
            f.write(patch)
        subprocess.run(["git", "apply", TEMP_PATCH], check=True, **DEVNULL)
        return patch
    except subprocess.CalledProcessError:
        return None
    finally:
        if os.path.exists(TEMP_PATCH):
            os.remove(TEMP_PATCH)
        subprocess.run(["git", "-C", ".", "reset", "--hard"], check=True, **DEVNULL)
        subprocess.run(["git", "clean", "-fdx"], check=True, **DEVNULL)
        os.chdir(cwd)


def extract_entities_from_directory(
    directory_path: str,
    entity_type: str,
    exclude_tests: bool = True,
    max_entities: int = -1,
) -> list[CodeEntity]:
    """
    Extracts entities (functions, classes, etc.) from Python files in a directory.
    Args:
        directory_path (str): Path to the directory to scan.
        entity_type (str): Type of entity to extract.
        exclude_tests (bool): Whether to exclude test files and directories.
    Returns:
        List[CodeEntity]: List of CodeEntity objects containing entity information.
    """
    entities = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if not file.endswith(".py"):
                continue
            if exclude_tests and any(
                [x in root for x in ["/tests", "/test", "/testing"]]
            ):
                continue
            if exclude_tests and file.startswith("test_"):
                continue

            file_path = os.path.join(root, file)

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    file_content = f.read()
            except:
                continue

            try:
                tree = ast.parse(file_content, filename=file_path)
            except SyntaxError:
                continue

            for node in ast.walk(tree):
                if not any([isinstance(node, x) for x in ENTITY_TYPES[entity_type]]):
                    continue
                entities.append(get_entity_from_node(node, file_content, file_path))
                if max_entities != -1 and len(entities) >= max_entities:
                    return entities
    return entities


def get_combos(items, r, max_combos) -> list[tuple]:
    """Get `max_combos` combinations of items of length r or greater."""
    all_combos = []
    for new_combo in combinations(items, r):
        all_combos.append(new_combo)
        if max_combos != -1 and len(all_combos) >= max_combos:
            break
    return sorted(all_combos, key=len)


def get_entity_from_node(
    node: ast.AST, file_content: str, file_path: str
) -> CodeEntity:
    """Turns an AST node into a CodeEntity object."""
    start_line = node.lineno  # type: ignore[attr-defined]
    end_line = (
        node.end_lineno if hasattr(node, "end_lineno") else None  # type: ignore[attr-defined]
    )

    if end_line is None:
        # Calculate end line manually if not available (older Python versions)
        end_line = (
            start_line
            + len(
                ast.get_source_segment(file_content, node).splitlines()  # type: ignore[attr-defined]
            )
            - 1
        )

    source_code = ast.get_source_segment(file_content, node)

    # Get the line content for the source definition
    source_line = file_content.splitlines()[start_line - 1]
    leading_whitespace = len(source_line) - len(source_line.lstrip())

    # Determine the number of spaces per tab
    indent_size = 4  # Default fallback
    if "\t" in file_content:
        indent_size = source_line.expandtabs().index(source_line.lstrip())

    # Calculate indentation level
    indentation_level = (
        leading_whitespace // indent_size if leading_whitespace > 0 else 0
    )

    # Remove indentation from source source code
    assert source_code is not None
    lines = source_code.splitlines()
    dedented_source_code = [lines[0]]
    for line in lines[1:]:
        # Strip leading spaces equal to indentation_level * indent_size
        dedented_source_code.append(line[indentation_level * indent_size :])
    source_code = "\n".join(dedented_source_code)

    return CodeEntity(
        file_path=file_path,
        indent_level=indentation_level,
        indent_size=indent_size,
        line_end=end_line,
        line_start=start_line,
        src_code=source_code,
        src_node=node,
    )


def get_patch(repo: str, reset_changes: bool = False):
    """Get the patch for the current changes in a Git repository."""

    if (
        not os.path.isdir(repo)
        or subprocess.run(["git", "-C", repo, "status"], **DEVNULL).returncode != 0
    ):
        raise FileNotFoundError(f"'{repo}' is not a valid Git repository.")

    # Stage every change in the working tree so the diff only contains staged files
    subprocess.run(["git", "-C", repo, "add", "-A"], check=True, **DEVNULL)

    # Capture the staged diff as a patch string
    patch = subprocess.run(
        ["git", "-C", repo, "diff", "--staged"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    if len(patch.strip()) == 0:
        # No changes were staged, so no patch is needed
        return None

    # Clean up the index and the working tree to avoid unwanted side‑effects
    for cleanup_cmd in [
        f"git -C {repo} restore --staged .",
        f"git -C {repo} reset --hard",
        f"git -C {repo} clean -fdx",
    ]:
        subprocess.run(cleanup_cmd.split(), check=True, **DEVNULL)

    # -------------------------------------------------------------
    # Apply the requested patch changes (diff from the staged state)
    # -------------------------------------------------------------

    # Write the patch outside the working tree so it never appears in `git add -A`
    patch_file = tempfile.NamedTemporaryFile(suffix=".diff", delete=False).name

    # Use explicit UTF‑8 encoding to avoid platform‑specific issues
    with open(patch_file, "w", encoding="utf-8") as f:
        f.write(patch)

    # Apply the diff back onto the working tree from the temporary file
    subprocess.run(["git", "-C", repo, "apply", patch_file], check=True)

    # Optionally reset the changes so the repository returns to a clean state
    if reset_changes:
        subprocess.run(["git", "-C", repo, "reset", "--hard"], check=True, **DEVNULL)
        subprocess.run(["git", "-C", repo, "clean", "-fdx"], check=True, **DEVNULL)

    return patch
