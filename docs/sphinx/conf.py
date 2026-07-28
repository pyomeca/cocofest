import importlib.util
import pathlib
import sys

repo_root = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

_version_spec = importlib.util.spec_from_file_location(
    "cocofest_version", repo_root / "cocofest" / "misc" / "__version__.py"
)
_version_module = importlib.util.module_from_spec(_version_spec)
_version_spec.loader.exec_module(_version_module)
release = _version_module.__version__

project = "Cocofest"
author = "Kev1Co"
copyright = "2026, Cocofest contributors"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinxcontrib.mermaid",
]

exclude_patterns = ["_build", "_templates", "_generated"]
templates_path = ["_templates"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

autodoc_mock_imports = ["bioptim", "biorbd", "casadi", "pyorerun"]

autosummary_generate = True
autodoc_default_options = {"undoc-members": True}
napoleon_google_docstring = False
napoleon_numpy_docstring = True

html_theme = "furo"
html_logo = "../assets/cocofest_logo.png"
html_static_path = ["_static"]
html_css_files = ["mermaid.css"]
html_js_files = ["center-mermaid.js"]


# Walks the cocofest package's top-level folders to build a clickable diagram,
# regenerated on every build so it never needs manual updates.
def _write_architecture_diagram(app):
    package_root = pathlib.Path(app.srcdir).resolve().parent.parent / "cocofest"
    lines = [
        '%%{init: {"themeVariables": {"fontSize": "16px"}}}%%',
        "flowchart TD",
        '    cocofest(["cocofest"])',
        '    click cocofest "api/index.html"',
    ]

    for entry in sorted(package_root.iterdir(), key=lambda p: p.name):
        if not entry.is_dir() or entry.name.startswith("_") or entry.name == "__pycache__":
            continue
        node_id = entry.name
        lines.append(f'    cocofest --> {node_id}["{entry.name}"]')
        lines.append(f'    click {node_id} "api/generated/cocofest.{entry.name}.html"')

    out_dir = pathlib.Path(app.srcdir) / "_generated"
    out_dir.mkdir(exist_ok=True)
    content = "```{mermaid}\n" + "\n".join(lines) + "\n```\n"
    (out_dir / "architecture.md").write_text(content, encoding="utf-8")


def setup(app):
    app.connect("builder-inited", _write_architecture_diagram)
