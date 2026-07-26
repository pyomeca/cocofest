import os
from pathlib import Path
import subprocess
import sys


def test_plot_module_respects_headless_matplotlib_backend():
    repository_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    environment["PYTHONPATH"] = str(repository_root)

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import cocofest.result.plot; "
                "import matplotlib; "
                "assert matplotlib.get_backend().lower() == 'agg'"
            ),
        ],
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
