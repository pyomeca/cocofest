# Message proposé pour Kevin

Salut Kevin,

J’ai adapté le benchmark des solveurs au nouveau problème de pédalage. Le
couple externe est maintenant une **assistance de 0,20 N·m** : dans la
convention du modèle, le couple signé vaut `-0.20 N.m` et fournit environ
`+1.2566 W` à la cadence nominale. L’objectif reste uniquement la minimisation
de la fatigue. Le problème commun à IPOPT, MadNLP et Alpaqa utilise une
collocation Radau de degré 3, un cycle par RHO, 30 stimulations par cycle et une
tolérance de `0.002 rad` sur l’angle terminal du pédalier.

Le workflow GitHub
`.github/workflows/cycling_solver_benchmark_linux.yml` prépare d’abord un seed
IPOPT assisté, vérifié physiquement, puis donne exactement le même artifact aux
trois solveurs. MadNLP et Alpaqa font ensuite un hot-start IPOPT périodique
certifié avant leur premier RHO. Le coût de cette préparation est publié
séparément : il ne faut pas le confondre avec le temps chaud des résolutions
suivantes.

Le run par défaut contient 30 RHO et utilise tous les cœurs visibles par
`nproc`, avec les pools BLAS/OpenMP/Julia internes limités à un thread. Deux
échecs consécutifs sont autorisés pour confirmer une non-convergence. Le
préfixe scientifique validé s’arrête néanmoins au premier RHO échoué ou
infaisable.

Le rapport final compare :

- la convergence globale, le nombre de cycles validés et le premier RHO
  échoué ;
- la durée mur-à-mur, la préparation, la somme des temps RHO, puis la médiane
  et le P90 des temps chauds ;
- pour chaque RHO, le statut générique et natif, les itérations, le temps
  solveur, le temps mur et l’infaisabilité primale recalculée à partir des
  bornes du NLP ;
- les patrons de largeur d’impulsion aux **cycles/RHO 10 et 30** du même run,
  en secondes et microsecondes, avec les bornes, la phase réelle du pédalier et
  sa vitesse. Ce ne sont pas deux OCP contenant respectivement 10 et 30 cycles.

La commande de lancement est :

```bash
gh workflow run cycling_solver_benchmark_linux.yml \
  --repo mickaelbegon/cocofest-pedalage \
  --ref codex/acados-pr-refresh \
  -f runner_label=ubuntu-22.04 \
  -f cycles=30 \
  -f crank_assistance_nm=0.20 \
  -f terminal_wheel_q_slack=0.002 \
  -f solver_max_iterations=2000
```

L’artifact `cycling-fatigue-kevin-report-*` contient le rapport Markdown, le
JSON combiné, `rho-timings.csv`, `stimulation-patterns.csv` et les logs
complets.

Points sensibles, en toute transparence :

- nous n’avons pas encore de résultat Linux 30 cycles fiable sur ce nouveau
  problème ; les anciens résultats locaux à 30 cycles utilisaient une fenêtre
  de deux cycles et une ancienne gestion du raccord, donc je les ai
  volontairement exclus ;
- les tests locaux de construction et de reporting passent (`167 passed`),
  mais le chemin qui compile CasADi avec Alpaqa dans GitHub Actions n’a pas
  encore été exécuté sur un runner Linux ;
- les intégrations MadNLP et Alpaqa sont pour l’instant sur deux commits
  Bioptim distincts. Les configurations OCP sont comparées automatiquement,
  et les deux SHA sont inscrits dans le rapport, mais cette différence de
  branche reste un facteur de confusion à garder en tête ;
- les essais locaux sans hot-start IPOPT n’étaient pas convergés
  (`inf_pr ≈ 1.58` pour MadNLP et `≈ 8.65` pour Alpaqa sur un cycle). Ils
  démontrent que les plugins s’exécutent, pas leurs performances. C’est
  précisément pourquoi le benchmark Linux impose maintenant le seed et le
  hot-start certifiés ;
- le patron de stimulation peut ne pas être unique : il n’y a ni
  régularisation des contrôles ni cadence intra-cycle prescrite, et
  l’assistance peut amener les muscles à réguler ou freiner. Une différence de
  patron doit donc être interprétée avec le coût de fatigue, `q`, `qdot` et la
  faisabilité, jamais seule ;
- IPOPT utilise MUMPS dans l’action publique. MA57 exige CoinHSL sous licence
  et reste réservé à un runner privé déjà configuré.

Pour l’instant, IPOPT reste la référence robuste. MadNLP est le candidat le
plus crédible à comparer après hot-start sur Linux. Alpaqa est intéressant
pour des résolutions répétées, mais sa sensibilité au scaling et l’absence de
statistiques ALM/PANOC détaillées dans le plugin CasADi rendent son diagnostic
plus délicat. Le run 30 RHO permettra enfin de juger ces deux solveurs sur des
temps chauds et une faisabilité homogène, plutôt que sur des smokes à froid.
