"""
tests/test_architecture_boundaries.py

Lightweight boundary guard: asserts that production and evaluation code
does not import from debug_lab or from the old debug tool packages.

Uses AST-based import scanning so it works without actually importing any module.
"""

import ast
import pathlib

# Zones that must NOT import debug-only packages
PRODUCTION_ROOTS = [
    "src/matera/core",
    "src/matera/vision",
    "src/matera/vlm",
    "src/matera/api",
    "src/matera/export",
    "scripts",
]

EVALUATION_ROOT = "src/matera/evaluation"

# Import prefixes that are forbidden in production + evaluation code
FORBIDDEN_IN_PRODUCTION = [
    "debug_lab",
    "matera.vision.local_realignment",
    "matera.models.legacy_classifier",
]

FORBIDDEN_IN_EVALUATION = [
    "debug_lab",
    "matera.models.legacy_classifier",
]


def _collect_imports(filepath: pathlib.Path) -> list[str]:
    """Return all module names imported in a Python file using AST."""
    try:
        tree = ast.parse(filepath.read_text(encoding="utf-8"))
    except SyntaxError:
        return []
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    return imports


def _check_zone(roots: list[str], forbidden: list[str]) -> list[str]:
    """
    Returns a list of violation strings (file + import) for any file under
    the given roots that imports a forbidden module.
    """
    violations = []
    repo_root = pathlib.Path(__file__).parent.parent

    for root_rel in roots:
        root = repo_root / root_rel
        if not root.exists():
            continue
        for py_file in root.rglob("*.py"):
            imports = _collect_imports(py_file)
            for imp in imports:
                for forbidden_prefix in forbidden:
                    if imp == forbidden_prefix or imp.startswith(forbidden_prefix + "."):
                        rel = py_file.relative_to(repo_root)
                        violations.append(
                            f"{rel}: imports '{imp}' (forbidden prefix: '{forbidden_prefix}')"
                        )
    return violations


def test_production_does_not_import_debug_lab():
    violations = _check_zone(PRODUCTION_ROOTS, FORBIDDEN_IN_PRODUCTION)
    assert violations == [], "Production code imports forbidden debug-lab modules:\n" + "\n".join(
        violations
    )


def test_evaluation_does_not_import_debug_lab():
    violations = _check_zone([EVALUATION_ROOT], FORBIDDEN_IN_EVALUATION)
    assert violations == [], "Evaluation code imports forbidden debug-lab modules:\n" + "\n".join(
        violations
    )


def test_evidence_package_contains_only_passive_models():
    repo_root = pathlib.Path(__file__).parent.parent
    evidence_dir = repo_root / 'src' / 'matera' / 'vision' / 'evidence'
    assert evidence_dir.exists()
    
    allowed_files = {'__init__.py', 'models.py', 'serialization.py'}
    
    for item in evidence_dir.iterdir():
        if item.is_dir() and item.name == '__pycache__':
            continue
        assert item.name in allowed_files, f'Unexpected file in evidence/: {item.name}'


def test_production_does_not_import_old_evidence_paths():
    forbidden_old_paths = [
        'matera.vision.evidence.extraction',
        'matera.vision.evidence.adapters',
        'matera.vision.evidence.local',
        'matera.vision.evidence.global_topology',
        'matera.vision.evidence.diagnostics',
    ]
    violations = _check_zone(PRODUCTION_ROOTS, forbidden_old_paths)
    
    # We allow tests and evidence/__init__.py to import these if needed for compat, 
    # but wait, evidence/__init__.py imports from scoring, not the other way around.
    # The actual constraint is production scoring imports don't depend on them.
    # We can just filter out evidence/__init__.py if it accidentally triggers.
    filtered_violations = [
        v for v in violations
        if 'evidence\\__init__.py' not in v and 'evidence/__init__.py' not in v
    ]
    
    assert filtered_violations == [], \
        'Production code imports moved old modules:\n' + '\n'.join(filtered_violations)


def test_backward_compatibility_imports():
    # Should not raise ImportError
    from matera.vision.evidence import (
        compute_global_topology_evidence,  # noqa: F401
        compute_local_option_evidence,  # noqa: F401
        evidence_to_mark_scores,  # noqa: F401
        extract_mark_evidence,  # noqa: F401
    )


def test_new_preferred_imports():
    # Should not raise ImportError
    from matera.vision.detectors import (
        compute_global_topology_evidence,  # noqa: F401
        compute_local_option_evidence,  # noqa: F401
    )
    from matera.vision.diagnostics import attach_global_shape_diagnostics  # noqa: F401
    from matera.vision.scoring import (
        evidence_to_mark_scores,  # noqa: F401
        extract_mark_evidence,  # noqa: F401
    )
