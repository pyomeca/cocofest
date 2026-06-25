# ACADOS Environment

The conda environment contains the compilers, CMake, CasADi and bioptim pieces needed by the cycling MHE, but ACADOS
also needs its native C libraries and the matching editable Python interface.

Build ACADOS and install the matching Python interface into the conda environment:

```bash
conda env update -f environment.yml --prune
scripts/setup_acados_for_conda_env.sh cocofest-run /path/to/acados
```

Before running an ACADOS solve, expose the native library location:

```bash
export ACADOS_SOURCE_DIR=/path/to/acados
export DYLD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib:${DYLD_LIBRARY_PATH:-}"  # macOS
export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib:${LD_LIBRARY_PATH:-}"      # Linux
```

The Python interface should come from the same ACADOS checkout as the compiled library:

```bash
conda run -n cocofest-run python -c "import acados_template; print(acados_template.__file__)"
```

Then the cycling MHE ACADOS example can be launched from the cycling examples directory:

```bash
cd examples/fes_multibody/cycling
conda run -n cocofest-run python cycling_mhe_acados.py
```

The pulse-width FES MHE still needs IPOPT in bioptim 3.4 because its stimulation and external-force numerical time
series are not currently exported as ACADOS parameters during code generation.
