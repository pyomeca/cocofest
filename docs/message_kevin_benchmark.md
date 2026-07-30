# Message proposé pour Kevin

Salut Kevin,

Le benchmark du RHO de pédalage est documenté ici :

[Benchmark des solveurs du RHO de pédalage FES](cycling_solver_benchmark/README.md)

Le problème minimise uniquement la fatigue, sans couple externe. Chaque OCP
contient un cycle de pédalage et 30 stimulations par muscle; une campagne de
100 RHO enchaîne donc 100 OCP d’un cycle. Les PW sont bornées dans
`[pd0, 600 µs]`, avec `pd0 ≈ 131.405 µs`, et la cible terminale d’angle est
absolue avec un slack de `0.002 rad`.
La référence absolue est recalée après le chargement du seed, y compris quand
le seed et le solveur utilisent tous deux la formulation full ou tous deux la
formulation reduced. Ce point corrige un décalage initial full de `3.94 mrad`
observé avec cinq statuts IPOPT pourtant égaux à zéro; le slack n’a pas été
relâché pour masquer cette incohérence.

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
du graphe. Chaque cas possède un répertoire de codegen isolé et la CI exige
une observation de ce source à chaque RHO, ce qui empêche un `nlp.c` full
résiduel de certifier artificiellement le reduced. L’ancienne ablation
intégrée qui reconstruisait deux OCP full
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

La campagne graduelle est terminée :

- [5 RHO, run 30519570984](https://github.com/mickaelbegon/cocofest/actions/runs/30519570984);
- [30 RHO, run 30520968417](https://github.com/mickaelbegon/cocofest/actions/runs/30520968417);
- [100 RHO, run 30522170340](https://github.com/mickaelbegon/cocofest/actions/runs/30522170340).

Le run final utilise Cocofest
`aac9ff5c2ccec2f16adb6fb1f46932d44e15b7f7` et le Bioptim commun
`3523f1745e315f07761159d7e06bd2d876026704`.

Sur 100 RHO, IPOPT et MadNLP convergent 100/100 en full et reduced. Les
médianes chaudes sont `1.444/1.122 s` pour IPOPT full/reduced et
`2.600/1.180 s` pour MadNLP. En revanche, MadNLP gagne le mur-à-mur :
`350.1/178.4 s` contre `612.4/255.6 s`. Pour IPOPT, le temps résiduel non
attribué aux solves ni à la préparation vaut environ `443 s` en full et
`123 s` en reduced sur une machine neuve. Ce n’est pas un chronométrage
instrumenté de la compilation; ce résidu est vraisemblablement dominé par la
génération/compilation, avec possiblement d’autres frais Python et système.
Une extrapolation des médianes chaudes place le rattrapage d’IPOPT compilé
face à MadNLP vers 330 RHO en full et 1580 RHO en reduced. Cela ne mesure pas
l’amortissement par rapport à IPOPT interprété, qui demanderait un cas témoin
strictement identique avec `compile=false`.

Les résultats de fatigue sont très proches entre solveurs à mécanique fixée.
En full, IPOPT/MadNLP donnent un coût `11406.6/11344.7`, une AUC
`16.489/16.460` et une capacité minimale `0.86736/0.86795`. En reduced, ces
valeurs sont `668.1/653.0`, `4.849/4.812` et `0.97649/0.97707`. Le grand écart
full/reduced vient principalement du Biceps et du deltoïde postérieur; la
réduction n’est donc pas encore physiologiquement interchangeable avec le
modèle full.

Les patrons IPOPT/MadNLP full sont presque identiques aux cycles 10 et 30.
En reduced, ils sont identiques au cycle 10 mais divergent localement au
cycle 30 sur le Biceps malgré des coûts proches : maximum `422.6 µs` avec
IPOPT contre `249.9 µs` avec MadNLP. ACADOS reduced atteint environ
`0.102 s/RHO`, mais son préfixe strict s’arrête à 13 cycles; il ne fournit
donc pas de patron valide au cycle 30. Toutes ses variantes full échouent
avant le premier cycle.

Les points qui restent sensibles sont la comparabilité physiologique de la
mécanique reduced, la robustesse ACADOS après plusieurs RHO et la
compatibilité structurelle Fatrop full/SX. Il ne faut donc pas interpréter un
gain de temps reduced comme un gain scientifique avant d’avoir comparé les
trajectoires mécaniques, les forces et les états Ding.
