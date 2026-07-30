# Message proposé pour Kevin

Salut Kevin,

Le benchmark du RHO de pédalage est documenté ici :

[Benchmark des solveurs du RHO de pédalage FES](cycling_solver_benchmark/README.md)

Le problème minimise uniquement la fatigue, sans couple externe. Chaque OCP
contient un cycle de pédalage et 30 stimulations par muscle; une campagne de
100 RHO enchaîne donc 100 OCP d’un cycle. Les PW sont bornées dans
`[pd0, 600 µs]`, avec `pd0 ≈ 131.405 µs`, et la cible terminale d’angle est
absolue avec un slack de `0.002 rad`.

La nouvelle campagne active compare :

- IPOPT/MUMPS full et reduced;
- MadNLP/MUMPS full et reduced;
- ACADOS full et reduced, avec ses variantes de restauration et RTI.

Tous les graphes de la campagne sont maintenant SX, y compris le warm-up
IPOPT. ACADOS consomme directement ce seed IPOPT/collocation SX certifié :
son raffinement IPOPT auxiliaire n’est plus répété dans chaque variante,
car deux runners ont été arrêtés après environ 150 s pendant sa construction
full redondante. Ce choix SX est mesuré, pas
seulement supposé : sur 30 RHO, SX a réduit la médiane chaude de 57.5 à
60.5 % par rapport à MX, soit une accélération de 2.35× à 2.53×. Les quatre
comparaisons IPOPT/MadNLP full/reduced ont convergé 30/30, avec un écart
maximal d’objectif compris entre `8.6e-12` et `5.01e-11`. Le runner refuse
désormais MX et le rapport vérifie `use_sx=true`.

MadNLP utilise uniquement MUMPS. Nous avons corrigé un point subtil de
libMad : l’API Cocofest accepte `mumps`, mais libMad attend le type exact
`MumpsSolver`. Avant la correction, le warning
`option linear_solver is of unknown type mumps, ignoring` signifiait que
l’option était ignorée; le calcul utilisait malgré tout le MUMPS par défaut.
Le smoke runtime et les logs font maintenant échouer la CI si ce warning
réapparaît.

PARDISO est volontairement laissé de côté. Il n’a pas apporté de gain :
dans le run `30511306081`, MUMPS était environ 30 % plus rapide en full et
6 % en reduced, avec aussi un meilleur temps mur-à-mur. Le commit libMad
épinglé contient encore le support PARDISO, mais le workflow ne le sélectionne
ni ne le certifie. Son smoke d’installation utilise `no_hsl_example`, qui
exerce MUMPS sans instancier PARDISO.

Fatrop est également sorti de la campagne active SX-only. Les anciens
résultats MX restent documentés, mais Fatrop full échoue encore en SX lors de
la détection de structure des gaps; reduced SX n’a validé qu’un smoke d’un
RHO. Il serait trompeur de le comparer en MX aux autres solveurs SX. Alpaqa
reste hors campagne parce que l’intégration actuelle ne valide pas le RHO.

La version Bioptim commune aux seeds, IPOPT, MadNLP et ACADOS est le SHA
`3523f1745e315f07761159d7e06bd2d876026704` du fork
`mickaelbegon/BiorbdOptim`. Chaque artefact exporte ce SHA et le SHA Cocofest.
MadNLP reste interprété : `--madnlp-c-compile` est reconnu par la CLI mais
Bioptim lève encore `NotImplementedError`. La compilation persistante testée
dans cette campagne concerne donc IPOPT.

La validation doit être graduelle : 5 RHO, puis 30, puis 100 uniquement si le
palier précédent produit tous les artefacts attendus, sans erreur
d’infrastructure, sans graphe non-SX et sans option libMad ignorée.

La compilation IPOPT est validée directement dans les cas full et reduced :
la bibliothèque doit être construite une seule fois, le source `nlp.c` doit
garder le même hash et les bornes mobiles doivent changer sans reconstruction
du graphe. L’ancienne ablation intégrée qui reconstruisait deux OCP full
supplémentaires a été retirée après un arrêt `143` du runner. Si tu veux
remesurer compilé contre interprété, il faut lancer deux workflows identiques
en changeant seulement `compile_nlp_evaluators`; les artefacts principaux
restent alors directement comparables.
Une non-convergence numérique reste bien sûr un résultat à analyser.

Commande type, avec `N=5`, puis `30`, puis `100` :

```bash
gh workflow run cycling_solver_benchmark_linux.yml \
  --repo mickaelbegon/cocofest \
  --ref codex/acados-pr-refresh \
  -f runner_label=ubuntu-24.04 \
  -f cycles=N \
  -f cycles_per_window=1 \
  -f crank_assistance_nm=0.00 \
  -f terminal_wheel_q_slack=0.002 \
  -f compile_nlp_evaluators=true \
  -f solver_max_iterations=2000 \
  -f seed_validation_max_iterations=2000 \
  -f acados_smoke_rhos=N \
  -f acados_option_rhos=N
```

Les artefacts donnent la convergence, le temps mur-à-mur, les temps chauds
par RHO, le coût de fatigue, la fatigue cumulée et la capacité finale des
quatre muscles, les patrons de stimulation aux cycles 10 et 30, ainsi que les
variations de PW entre cycles.

Les points qui restent sensibles sont la comparabilité physiologique de la
mécanique reduced, la robustesse ACADOS après plusieurs RHO et la
compatibilité structurelle Fatrop full/SX. Il ne faut donc pas interpréter un
gain de temps reduced comme un gain scientifique avant d’avoir comparé les
trajectoires mécaniques, les forces et les états Ding.
