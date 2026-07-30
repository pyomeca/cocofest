# Message proposé pour Kevin

Salut Kevin,

Le benchmark du RHO de pédalage est maintenant documenté dans un README
unique :

[Benchmark des solveurs du RHO de pédalage FES](cycling_solver_benchmark/README.md)

Le problème courant minimise uniquement la fatigue, avec un couple externe
nul, un cycle et 30 stimulations par muscle dans chaque OCP. Le benchmark
enchaîne 100 RHO et compare IPOPT/MUMPS, MadNLP/MUMPS,
MadNLP/PARDISO-MKL, Fatrop/collocation et ACADOS, sur les mécaniques full et
reduced.

Les points importants avant de lancer sur le calculateur sont :

- IPOPT, MadNLP et Fatrop/collocation ont validé 100/100 RHO dans le dernier
  run Linux;
- IPOPT/MUMPS full reste la référence robuste;
- MadNLP/MUMPS reduced était le NLP le plus rapide du dernier run, mais ce
  classement doit être remesuré avec le nouvel appariement mécanique;
- PARDISO n’a pas battu MUMPS sur le runner actuel;
- ACADOS est très rapide lorsqu’il converge, mais perd encore la faisabilité
  après environ huit RHO dans le cas reduced;
- Alpaqa est laissé de côté parce que l’intégration actuelle ne valide aucun
  RHO;
- RK4 a été retiré du benchmark Fatrop;
- les PW sont bornées dans `[pd0, 600 µs]`, avec
  `pd0 ≈ 131.405 µs`;
- la cible angulaire est absolue, avec une tolérance terminale de `0.002 rad`.

Le grand écart de fatigue historique entre full et reduced ne doit plus être
utilisé. Le nouveau benchmark résout d’abord le seed reduced, puis relève
exactement `q(theta)` et `qdot=T(theta) omega` pour initialiser le full. Les
contraintes de contact sont imposées au début de chaque RHO et l’angle ainsi
que la cadence sont audités physiquement, sans assimiler `q[2]` à l’angle du
pédalier.

Sur un smoke test local IPOPT/MUMPS à `0 N.m`, un seul RHO, le coût exécuté
est passé à `3.599121` en reduced contre `3.604350` en full, soit environ
`0.15 %` d’écart. C’est très encourageant, mais pas encore une validation
30/100 RHO sur Linux.

Points encore sensibles :

- la contrainte terminale physique cross/dot a fait avorter l’initialisation
  IPOPT locale; elle est désactivée et remplacée provisoirement par une borne
  `q[2]` resserrée puis un audit absolu de `theta`;
- ACADOS teste maintenant une restauration native
  `FEASIBILITY_QP -> SQP nominal`, mais le résultat long n’est pas encore
  disponible;
- Fatrop reste sans scaling d’état tant que les gaps normalisés
  `z[k+1] - S^-1 Phi(S z[k],u[k])` ne sont pas intégrés dans Bioptim;
- les ablations interprété/compilé IPOPT, MadNLP et Fatrop sont nouvelles et
  doivent être confirmées par le prochain CI; les campagnes principales
  full/reduced utilisent également la compilation persistante;

Pour lancer le benchmark :

```bash
gh workflow run cycling_solver_benchmark_linux.yml \
  --repo mickaelbegon/cocofest \
  --ref codex/acados-pr-refresh \
  -f runner_label=ubuntu-24.04 \
  -f cycles=100 \
  -f cycles_per_window=1 \
  -f crank_assistance_nm=0.00 \
  -f terminal_wheel_q_slack=0.002 \
  -f compile_nlp_evaluators=true \
  -f solver_max_iterations=2000 \
  -f seed_validation_max_iterations=2000 \
  -f acados_smoke_rhos=100 \
  -f acados_option_rhos=5
```

Les artefacts contiennent le temps mur-à-mur, les temps chauds par RHO, la
faisabilité, la fatigue des quatre muscles, les patrons aux cycles 10 et 30,
les variations de PW et la décomposition Hessien/Jacobien de Fatrop.
Ils incluent aussi les résidus de projection mécanique, les traces physiques
`theta/omega`, le hash persistant de `nlp.c` et l’ablation de compilation sur
cinq RHO.
