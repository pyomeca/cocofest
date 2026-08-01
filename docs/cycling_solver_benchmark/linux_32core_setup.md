# Installation et benchmark sur Linux 32 cœurs

Cette procédure reproduit les piles numériques du workflow
[`cycling_solver_benchmark_linux.yml`](../../.github/workflows/cycling_solver_benchmark_linux.yml)
sur une machine x86-64 dédiée.

## 1. Architecture recommandée

Utiliser deux environnements Conda construits depuis le même fichier de base :

| Environnement | Solveurs | CasADi | ABI C++ |
|---|---|---|---:|
| `cocofest-rho32` | IPOPT, FATROP, ACADOS | roue officielle `3.7.2` | `0` |
| `cocofest-madnlp32` | MadNLP/MUMPS et IPOPT de préparation | source `3.7.2` avec libMad | `1` |

Il ne faut pas installer la roue CasADi officielle dans l'environnement
MadNLP après sa compilation : elle écraserait le plugin et pourrait rendre
biorbd-CasADi incompatible. De même, biorbd-CasADi doit être recompilé dans
chaque environnement avec l'ABI de son CasADi.

## 2. Versions épinglées

| Composant | Version ou commit |
|---|---|
| Ubuntu | `24.04` recommandé |
| Python | `3.11` |
| CasADi officiel | `3.7.2` |
| CasADi MadNLP | `973b086f4dcda9f49cd9c1948432ae4b7ee54886` |
| Bioptim | `dad96b90d47c36126c1e97ec35f27c499abf4b12` |
| Branche Bioptim | `codex/cocofest-acados-v055-exploration` |
| ACADOS | `59d93e17d2985fdd73fc58b8a83ed8f83a024171` |
| libMad | `5529f23a6bff33c566ad954da38d352f1f172356` |
| Julia | `1.12.6` |
| JuliaC | `73be8587a80bbb65dab7acd71d406f72867a3571` |
| biorbd | `Release_1.12.2` |
| RBDL-CasADi | `93475e2ea9bc87f37709a2312533ce3187f054b9` |

La branche MadNLP exige un `libgcc_s.so.1` exportant `GCC_13.0.0`. Ubuntu
24.04 satisfait le contrat utilisé par la CI. Une distribution plus ancienne
doit réussir le script `check_libmad_host_linux.sh` avant toute compilation.

## 3. Dimensionnement de la machine

- x86-64 Linux;
- 32 CPU disponibles dans le cpuset du processus;
- au moins 32 Gio de RAM pour le benchmark RHO; 64 Gio donnent davantage de
  marge pour les compilations simultanées;
- environ 40 Gio libres pour Miniforge, Julia, les sources, builds et
  artefacts;
- accès réseau à GitHub, conda-forge, PyPI et aux serveurs Julia.

Le sweep full horizon n'est pas inclus dans cette estimation. Son problème
actuel est numérique avant d'être mémoire; une machine 128 Gio ne le corrige
pas sans meilleure seed multi-cycle.

## 4. Préparer Ubuntu

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  build-essential ca-certificates curl git git-lfs jq \
  librhash-dev ninja-build pkg-config tar unzip wget xz-utils

git lfs install

uname -a
uname -m
lscpu
nproc
free -h
df -h
cc --version
```

Vérifier que `nproc` retourne 32 dans le shell qui lancera le benchmark. Sur
un ordonnanceur, `nproc --all` peut afficher la machine entière tandis que
`nproc` reflète correctement le cpuset alloué.

## 5. Installer Miniforge

Si Conda/Miniforge est déjà installé, ignorer cette section et vérifier que
`conda config --show channels` utilise uniquement `conda-forge`.

```bash
export MINIFORGE_PREFIX="${HOME}/miniforge3"

curl -fsSLo /tmp/Miniforge3.sh \
  "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"

bash /tmp/Miniforge3.sh -b -p "$MINIFORGE_PREFIX"
source "$MINIFORGE_PREFIX/etc/profile.d/conda.sh"

conda config --set channel_priority strict
conda config --remove-key channels 2>/dev/null || true
conda config --add channels conda-forge
conda --version
```

Ajouter la ligne suivante au fichier d'initialisation du shell si nécessaire :

```bash
source "${HOME}/miniforge3/etc/profile.d/conda.sh"
```

## 6. Cloner les sources épinglées

Choisir un chemin absolu accessible par l'utilisateur du benchmark :

```bash
export RHO_WORK_ROOT="/chemin/absolu/vers/rho-work"
mkdir -p "$RHO_WORK_ROOT"
cd "$RHO_WORK_ROOT"

git clone --branch codex/acados-pr-refresh \
  https://github.com/mickaelbegon/cocofest.git cocofest
export COCOFEST_ROOT="$RHO_WORK_ROOT/cocofest"

mkdir -p "$COCOFEST_ROOT/.benchmark-deps"

git clone --recurse-submodules \
  https://github.com/mickaelbegon/BiorbdOptim.git \
  "$COCOFEST_ROOT/.benchmark-deps/bioptim"
git -C "$COCOFEST_ROOT/.benchmark-deps/bioptim" checkout \
  dad96b90d47c36126c1e97ec35f27c499abf4b12
git -C "$COCOFEST_ROOT/.benchmark-deps/bioptim" submodule update \
  --init --recursive

git clone https://github.com/mickaelbegon/libMad.git \
  "$COCOFEST_ROOT/.benchmark-deps/libMad"
git -C "$COCOFEST_ROOT/.benchmark-deps/libMad" checkout \
  5529f23a6bff33c566ad954da38d352f1f172356

test "$(git -C "$COCOFEST_ROOT/.benchmark-deps/bioptim" rev-parse HEAD)" = \
  dad96b90d47c36126c1e97ec35f27c499abf4b12
test "$(git -C "$COCOFEST_ROOT/.benchmark-deps/bioptim/external/acados" rev-parse HEAD)" = \
  59d93e17d2985fdd73fc58b8a83ed8f83a024171
test "$(git -C "$COCOFEST_ROOT/.benchmark-deps/libMad" rev-parse HEAD)" = \
  5529f23a6bff33c566ad954da38d352f1f172356
```

Avant ce clone, les changements locaux doivent être commités et poussés. Une
nouvelle machine ne peut pas récupérer les fichiers non suivis ou non poussés.

## 7. Environnement IPOPT, FATROP et ACADOS

### 7.1 Créer l'environnement

```bash
source "${HOME}/miniforge3/etc/profile.d/conda.sh"
cd "$COCOFEST_ROOT"

conda env create \
  --name cocofest-rho32 \
  --file .github/cycling-benchmark-linux-environment.yml
conda activate cocofest-rho32

export CMAKE_BUILD_PARALLEL_LEVEL=32
export CASADI_CXX_ABI=0
export PYTHONPATH="$COCOFEST_ROOT${PYTHONPATH:+:$PYTHONPATH}"

python -m pip install --no-deps "casadi==3.7.2"
bash .github/scripts/install_biorbd_casadi_linux.sh
python -m pip install --no-deps -e .benchmark-deps/bioptim
```

Si l'environnement existe déjà, utiliser :

```bash
conda env update \
  --name cocofest-rho32 \
  --file .github/cycling-benchmark-linux-environment.yml \
  --prune
```

### 7.2 Compiler ACADOS 0.5.5

```bash
conda activate cocofest-rho32
cd "$COCOFEST_ROOT"

export CMAKE_BUILD_PARALLEL_LEVEL=32
bash .benchmark-deps/bioptim/external/acados_install_linux.sh \
  32 "$CONDA_PREFIX"

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Le script installe `acados_template` dans l'environnement et configure son
préfixe sur `$CONDA_PREFIX`. Il modifie temporairement le sous-module ACADOS,
puis restaure les fichiers suivis à la fin; ne conserver aucun développement
non commité dans ce sous-module pendant l'installation.

### 7.3 Valider l'environnement

```bash
conda activate cocofest-rho32
cd "$COCOFEST_ROOT"
export PYTHONPATH="$COCOFEST_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

test -f "$CONDA_PREFIX/lib/libacados.so"
test -f "$CONDA_PREFIX/lib/libhpipm.so"

python - <<'PY'
import acados_template
import biorbd_casadi
import casadi as cas

print("CasADi", cas.__version__)
print("biorbd", biorbd_casadi.__version__)
print("acados_template", acados_template.__file__)
assert cas.has_nlpsol("ipopt")
assert cas.has_nlpsol("fatrop")
PY

python -m pytest -q \
  tests/shard1/test_solver_backends.py \
  tests/shard1/test_reduced_cycling.py \
  tests/test_benchmark_readme.py
```

## 8. Environnement MadNLP/MUMPS

### 8.1 Installer Julia 1.12.6

```bash
curl -fsSL https://install.julialang.org | \
  sh -s -- --yes --default-channel 1.12.6

export PATH="${HOME}/.juliaup/bin:${HOME}/.julia/bin:$PATH"
juliaup status
julia --version
```

### 8.2 Vérifier le runtime hôte

```bash
cd "$COCOFEST_ROOT"
bash .github/scripts/check_libmad_host_linux.sh
```

Ne pas continuer si le script ne trouve pas `GCC_13.0.0` dans la bibliothèque
`libgcc_s.so.1` réellement résolue par `cc`.

### 8.3 Créer et compiler la pile MadNLP

```bash
source "${HOME}/miniforge3/etc/profile.d/conda.sh"
cd "$COCOFEST_ROOT"

conda env create \
  --name cocofest-madnlp32 \
  --file .github/cycling-benchmark-linux-environment.yml
conda activate cocofest-madnlp32

export PATH="${HOME}/.juliaup/bin:${HOME}/.julia/bin:$PATH"
export CMAKE_BUILD_PARALLEL_LEVEL=32
export CASADI_VERSION=3.7.2
export CASADI_MADNLP_COMMIT=973b086f4dcda9f49cd9c1948432ae4b7ee54886
export CASADI_CXX_ABI=1
export JULIAC_COMMIT=73be8587a80bbb65dab7acd71d406f72867a3571
export PYTHONPATH="$COCOFEST_ROOT${PYTHONPATH:+:$PYTHONPATH}"

bash .github/scripts/install_libmad_mumps_linux.sh \
  .benchmark-deps/libMad \
  .cache/madnlp-mumps \
  "$JULIAC_COMMIT"

export LD_LIBRARY_PATH="$COCOFEST_ROOT/.cache/madnlp-mumps/lib:$COCOFEST_ROOT/.cache/madnlp-mumps/share/julia/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

bash .github/scripts/install_casadi_madnlp_linux.sh \
  .cache/madnlp-mumps
bash .github/scripts/install_biorbd_casadi_linux.sh
python -m pip install --no-deps -e .benchmark-deps/bioptim
```

Si l'environnement existe déjà, remplacer `conda env create` par
`conda env update --name cocofest-madnlp32 --file ... --prune`, puis
recompiler CasADi et biorbd seulement si le commit, l'ABI ou une dépendance a
changé.

### 8.4 Valider MadNLP et MUMPS

```bash
conda activate cocofest-madnlp32
cd "$COCOFEST_ROOT"

export PATH="${HOME}/.juliaup/bin:${HOME}/.julia/bin:$PATH"
export PYTHONPATH="$COCOFEST_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$COCOFEST_ROOT/.cache/madnlp-mumps/lib:$COCOFEST_ROOT/.cache/madnlp-mumps/share/julia/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python - <<'PY'
import casadi as cas
import bioptim
import biorbd_casadi

print("CasADi", cas.__version__)
print("flags", cas.CasadiMeta.compiler_flags())
print("Bioptim", bioptim.__version__)
print("biorbd", biorbd_casadi.__version__)
assert "-DCASADI_WITH_THREAD" in cas.CasadiMeta.compiler_flags()
assert cas.has_nlpsol("ipopt")
assert cas.has_nlpsol("madnlp")
assert hasattr(bioptim.Solver, "MADNLP")
PY

python -m pytest -q \
  tests/shard1/test_solver_backends.py \
  tests/shard1/test_reduced_cycling.py \
  tests/test_benchmark_readme.py
```

Pendant le premier solve MadNLP, vérifier l'absence du message :

```text
libMAD WARNING: option linear_solver is of unknown type mumps, ignoring
```

Le JSON et le log doivent indiquer le type exact `MumpsSolver`.

## 9. Politique des 32 cœurs

### 9.1 Baseline reproductible

Dans les deux environnements :

```bash
export BENCHMARK_THREADS=32
export CMAKE_BUILD_PARALLEL_LEVEL=32

export OMP_NUM_THREADS=1
export OMP_THREAD_LIMIT=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export JULIA_NUM_THREADS=1
```

`BENCHMARK_THREADS=32` autorise la construction parallèle prévue par le code.
Les variables numériques à 1 empêchent une oversubscription cachée et rendent
les temps comparables à la CI historique.

Pour ACADOS seulement, une campagne séparée peut utiliser :

```bash
export OMP_NUM_THREADS=32
export OMP_THREAD_LIMIT=32
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
```

Ne pas mélanger ces temps avec la baseline mono-thread numérique.

### 9.2 Sweep de parallélisme MUMPS

Après certification de la baseline, tester successivement
`1, 2, 4, 8, 16, 30, 32` threads. Exécuter au moins trois répétitions du même
NLP chaud et ne lancer aucun autre solveur en parallèle. Enregistrer :

- temps total;
- temps d'évaluation des fonctions;
- temps Jacobien/Hessienne;
- factorisation et backsolve;
- itérations;
- fréquence CPU et topologie NUMA.

Si le MUMPS fourni avec IPOPT ou libMad est sériel, modifier
`OMP_NUM_THREADS` ne donnera pas de gain. Il faut le constater dans les temps,
pas supposer un speedup de `32x`.

## 10. Construire la seed commune localement

Effectuer cette étape dans `cocofest-rho32`, avec les threads numériques à 1.

```bash
conda activate cocofest-rho32
cd "$COCOFEST_ROOT"

export GITHUB_WORKSPACE="$COCOFEST_ROOT"
export PYTHONPATH="$COCOFEST_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export BENCHMARK_THREADS=32
export OMP_NUM_THREADS=1
export OMP_THREAD_LIMIT=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p benchmark-seed-result benchmark-seed

prepare_seed() {
  mechanics="$1"
  mechanics_options=()
  if [ "$mechanics" = reduced ]; then
    mechanics_options+=(--mechanical-formulation reduced)
  else
    mechanics_options+=(
      --common-initial-solution "$COCOFEST_ROOT/benchmark-seed-result/common-reduced.npz"
    )
  fi

  python examples/fes_multibody/cycling/cycling_fes_solver_comparison.py \
    --solvers ipopt \
    --objective fatigue \
    --ipopt-profile periodic_collocation \
    --ipopt-use-sx \
    --ipopt-enforce-start-constraints \
    --cycles-per-window 1 \
    --stimulations-per-cycle 30 \
    --n-windows 1 \
    --n-threads "$BENCHMARK_THREADS" \
    --crank-assistance 0.00 \
    --standard-warmup-seed .github/benchmark-seeds/legacy-resistive-0p22-warmup.npz \
    --legacy-standard-warmup-seed-signed-torque 0.22 \
    --standard-warmup-seed-continuation \
    --warmup-ipopt-linear-solver mumps \
    --ipopt-linear-solver mumps \
    --ipopt-max-iter 2000 \
    --ipopt-disable-historical-initial-guess \
    --reduced-cycling-profile "$COCOFEST_ROOT/benchmark-seed-result/reduced-cycling-fourier12.npz" \
    --state-scaling full \
    --first-node-wheel-q-slack 0 \
    --terminal-wheel-q-slack 0.002 \
    --compact-rho-output \
    --print-traces \
    --common-initial-solution-output "$COCOFEST_ROOT/benchmark-seed-result/common-${mechanics}.npz" \
    --output-json "$COCOFEST_ROOT/benchmark-seed-result/seed-check-${mechanics}.json" \
    "${mechanics_options[@]}"

  jq -e '.results[0] | (.success == true and .attempted_windows == 1)' \
    "benchmark-seed-result/seed-check-${mechanics}.json"
}

prepare_seed reduced
prepare_seed full

cp benchmark-seed-result/common-reduced.npz benchmark-seed/
cp benchmark-seed-result/common-full.npz benchmark-seed/
cp benchmark-seed-result/reduced-cycling-fourier12.npz benchmark-seed/
```

Ne jamais réutiliser une seed provenant d'un autre couple, d'une autre force
passive ou d'une autre transcription sans validation explicite.

## 11. Lancer les premiers cas localement

Variables communes :

```bash
export GITHUB_WORKSPACE="$COCOFEST_ROOT"
export PYTHONPATH="$COCOFEST_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export BENCHMARK_THREADS=32
export BENCHMARK_CYCLES_PER_WINDOW=1
export BENCHMARK_ASSISTANCE=0.00
export BENCHMARK_Q_SLACK=0.002
export BENCHMARK_MAX_ITER=2000
export BENCHMARK_CYCLES=1
```

### 11.1 IPOPT et FATROP

```bash
conda activate cocofest-rho32
cd "$COCOFEST_ROOT"

bash .github/scripts/run_cycling_benchmark_case.sh \
  ipopt ipopt reduced mumps collocation local-results \
  "$BENCHMARK_CYCLES" true sx none 3

bash .github/scripts/run_cycling_benchmark_case.sh \
  ipopt-radau5 ipopt reduced mumps collocation local-results \
  "$BENCHMARK_CYCLES" false sx none 5

bash .github/scripts/run_cycling_benchmark_case.sh \
  fatrop-collocation fatrop reduced fatrop collocation local-results \
  "$BENCHMARK_CYCLES" true sx none 3
```

### 11.2 MadNLP/MUMPS

```bash
conda activate cocofest-madnlp32
cd "$COCOFEST_ROOT"

export PATH="${HOME}/.juliaup/bin:${HOME}/.julia/bin:$PATH"
export LD_LIBRARY_PATH="$COCOFEST_ROOT/.cache/madnlp-mumps/lib:$COCOFEST_ROOT/.cache/madnlp-mumps/share/julia/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

bash .github/scripts/run_cycling_benchmark_case.sh \
  madnlp-mumps madnlp reduced mumps collocation local-results \
  "$BENCHMARK_CYCLES" true sx none 3

bash .github/scripts/run_cycling_benchmark_case.sh \
  madnlp-mumps-radau5 madnlp reduced mumps collocation local-results \
  "$BENCHMARK_CYCLES" false sx none 5
```

Le script conserve une non-convergence dans `result.json` et peut retourner
un code shell nul pour permettre la suite de la campagne. Il faut donc lire le
JSON et le log; le code de sortie seul ne certifie pas le RHO.

Passer ensuite manuellement `BENCHMARK_CYCLES` à `5`, puis `30`, puis `100`.
Ne pas automatiser les trois valeurs dans une boucle : inspecter le préfixe
physique et les artefacts avant chaque palier.

## 12. Résumer les résultats locaux

```bash
conda activate cocofest-rho32
cd "$COCOFEST_ROOT"

python .github/scripts/summarize_cycling_benchmark.py \
  local-results/*/result.json \
  --output-dir local-summary
```

Contrôler au minimum dans chaque JSON :

- `success` et statut natif;
- premier RHO en échec;
- préfixe physique strict;
- durée de construction, compilation, préparation et solveur;
- fatigue, AUC et capacité par muscle;
- transcription, SX, force passive et backend linéaire;
- hash de la bibliothèque compilée.

## 13. Utiliser la machine comme runner GitHub Actions

Pour reproduire automatiquement la matrice et les artefacts, enregistrer un
seul runner avec le label personnalisé `linux-32core` :

1. ouvrir le dépôt GitHub;
2. aller dans `Settings > Actions > Runners`;
3. choisir `New self-hosted runner`, Linux x64;
4. exécuter sur la machine les commandes de téléchargement affichées par
   GitHub;
5. ajouter le label lors de la configuration :

```bash
./config.sh \
  --url https://github.com/mickaelbegon/cocofest \
  --token TOKEN_TEMPORAIRE_AFFICHÉ_PAR_GITHUB \
  --labels linux-32core
```

Ne jamais écrire le token dans le dépôt ou dans ce document. Il expire
rapidement. L'utilisateur du runner doit pouvoir exécuter les commandes
`sudo apt-get` du workflow sans invite interactive, ou les dépendances système
doivent être préinstallées et le workflow adapté.

Sur un dépôt public, ne pas autoriser du code de pull request non approuvé à
s'exécuter sur ce runner. Utiliser un runner ou un groupe limité à ce dépôt et
aux branches de confiance.

Avec une seule instance de runner sur la machine, les jobs seront séquentiels;
c'est souhaitable pour éviter la contention lors des mesures. Enregistrer
plusieurs instances sur la même machine rendrait les temps solveurs difficiles
à interpréter.

## 14. Campagne GitHub graduelle sur le runner 32 cœurs

Les modifications locales doivent d'abord être commités et poussées sur
`codex/acados-pr-refresh`.

### Gate 5 RHO

```bash
gh workflow run cycling_solver_benchmark_linux.yml \
  --repo mickaelbegon/cocofest \
  --ref codex/acados-pr-refresh \
  -f runner_label=linux-32core \
  -f cycles=5 \
  -f cycles_per_window=1 \
  -f crank_assistance_nm=0.00 \
  -f terminal_wheel_q_slack=0.002 \
  -f solver_max_iterations=2000 \
  -f compile_nlp_evaluators=true \
  -f acados_smoke_rhos=5 \
  -f acados_option_rhos=5 \
  -f refined_collocation_validation=true \
  -f refined_collocation_rhos=5
```

### Gate 30 RHO

Seulement après certification du gate 5 :

```bash
gh workflow run cycling_solver_benchmark_linux.yml \
  --repo mickaelbegon/cocofest \
  --ref codex/acados-pr-refresh \
  -f runner_label=linux-32core \
  -f cycles=30 \
  -f cycles_per_window=1 \
  -f crank_assistance_nm=0.00 \
  -f terminal_wheel_q_slack=0.002 \
  -f solver_max_iterations=2000 \
  -f compile_nlp_evaluators=true \
  -f acados_smoke_rhos=30 \
  -f acados_option_rhos=5 \
  -f refined_collocation_validation=true \
  -f refined_collocation_rhos=30
```

### Gate 100 RHO

Seulement après certification du gate 30 :

```bash
gh workflow run cycling_solver_benchmark_linux.yml \
  --repo mickaelbegon/cocofest \
  --ref codex/acados-pr-refresh \
  -f runner_label=linux-32core \
  -f cycles=100 \
  -f cycles_per_window=1 \
  -f crank_assistance_nm=0.00 \
  -f terminal_wheel_q_slack=0.002 \
  -f solver_max_iterations=2000 \
  -f compile_nlp_evaluators=true \
  -f acados_smoke_rhos=100 \
  -f acados_option_rhos=5 \
  -f refined_collocation_validation=true \
  -f refined_collocation_rhos=100
```

Après chaque lancement :

```bash
gh run list \
  --repo mickaelbegon/cocofest \
  --workflow cycling_solver_benchmark_linux.yml \
  --limit 5
```

Lire tous les logs et télécharger les artefacts avant d'ouvrir le gate suivant.
Un workflow vert signifie que l'infrastructure a terminé; il ne remplace pas
la lecture du préfixe physique de chaque solveur.

## 15. Ordre recommandé de la première journée

1. Préflight Ubuntu et CPU.
2. Installation de `cocofest-rho32` et smoke tests IPOPT/FATROP/ACADOS.
3. Construction de la seed commune.
4. Installation de `cocofest-madnlp32` et smoke test `MumpsSolver`.
5. Un RHO reduced Radau 3 et Radau 5 avec IPOPT et MadNLP.
6. Gate CI 5 RHO sur le runner `linux-32core`.
7. Inspection des JSON, du calcium, de la force passive et des temps.
8. Gate 30 seulement si le gate 5 est scientifiquement certifié.

La première décision expérimentale est de vérifier si Radau 5 atteint le
critère de convergence du calcium. La campagne 100 RHO et le sweep de threads
viennent après cette réponse, pas avant.

## 16. Références d'installation

- [Miniforge — installateurs et installation Linux](https://github.com/conda-forge/miniforge/blob/main/README.md)
- [Juliaup — installation et sélection d'une version Julia](https://github.com/JuliaLang/juliaup)
- [GitHub — ajouter un runner autohébergé](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/add-runners)
- [GitHub — labels des runners autohébergés](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/apply-labels)
