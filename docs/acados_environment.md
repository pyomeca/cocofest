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

Then launch the assisted cycling MHE from the repository root:

```bash
conda run -n cocofest-run python \
  examples/fes_multibody/cycling/cycling_pulse_width_mhe_acados_periodic.py \
  --crank-assistance 0.2 \
  --compact-rho-output
```

The example exports a one-cycle periodic-node FES formulation to ACADOS. IPOPT
remains part of its initialization pipeline: it constructs an assisted
standard collocation warmup and certifies a one-cycle periodic seed before the
timed ACADOS windows. Build, code-generation, and warm-start preparation times
are reported separately from the ACADOS window solve times.

The Linux benchmark workflow performs the equivalent installation from the
ACADOS submodule pinned by Bioptim, installs it into `$CONDA_PREFIX`, and
caches the complete native/Python stack. Its separate `acados-smoke` matrix
tests five full-mechanics and five experimental reduced-mechanics RHO by
default. Increase the workflow input `acados_smoke_rhos` to 30 only after the
short SQP jobs are stable; RTI is deliberately excluded from this first
validation. Set `cycles=acados` to run only seed preparation, the cached
ACADOS stack preparation, and these two smoke jobs.
