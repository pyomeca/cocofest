# Installation

`Cocofest` is not yet available on Anaconda/PyPI, so it must be installed from source:

```bash
git clone https://github.com/pyomeca/cocofest.git
cd cocofest
conda env create -f environment.yml
conda activate cocofest
pip install -e .
```

See the [README](https://github.com/pyomeca/cocofest#installation) for manual installation steps.

To build this documentation locally:

```bash
pip install -e ".[docs]"
sphinx-build -b html docs/sphinx docs/sphinx/_build/html
```
