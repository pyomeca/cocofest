# Message proposé pour Kevin

Salut Kevin,

Nous avons repris le benchmark du RHO de pédalage en corrigeant plusieurs
incohérences qui rendaient l’ancienne comparaison full/reduced difficile à
interpréter. Le rapport détaillé est ici :

[Benchmark des solveurs du RHO de pédalage FES](cycling_solver_benchmark/README.md)

Le problème minimise uniquement la fatigue, avec un couple externe nul. Chaque
OCP contient un cycle de pédalage et 30 stimulations par muscle. Les PW sont
bornées dans `[pd0, 600 µs]`, avec `pd0 ≈ 131.405 µs`; les anciens seeds sont
validés et tronqués dans cette plage au chargement. La cible terminale d’angle
est absolue, avec un slack de `0.002 rad`, afin d’éviter toute dérive cumulative
du pédalier.

Les changements importants sont :

- continuité exacte entre deux RHO de la cadence et des 20 états Ding;
- seed reduced commun, relevé sur la variété de contact pour initialiser le
  full au même instant physiologique;
- même loi de force passive en full et reduced;
- bornes de cadence exprimées dans les coordonnées physiques
  `theta/omega`, et non directement sur `qdot[2]`;
- audit indépendant de la variété de contact, de la vitesse tangentielle et de
  la progression angulaire;
- séparation entre RHO convergés isolément et préfixe strict réellement
  exécutable;
- graphes SX seulement, car ils sont nettement plus rapides que MX sur ce
  problème.

La branche Bioptim dédiée est
`codex/cocofest-acados-v055-exploration`, au SHA
`4179bf076b724fe6c4702739b3462e29ae4adef4`. Elle utilise ACADOS 0.5.5 au
SHA `59d93e17d2985fdd73fc58b8a83ed8f83a024171` et contient les correctifs
Bioptim pour les contraintes `Node.START`, le scaling ACADOS et le scaling
FATROP. La branche Cocofest est `codex/acados-pr-refresh`.

Le benchmark actif compare :

- IPOPT/MUMPS, full interprété et reduced compilé;
- MadNLP/MUMPS, full et reduced interprétés;
- FATROP/collocation compilé, full et reduced;
- ACADOS 0.5.5, SQP/IRK, RTI et variantes de restauration.

MadNLP utilise explicitement `MumpsSolver`. PARDISO a été laissé de côté, car
il n’a pas apporté de gain reproductible. Alpaqa est également hors campagne :
l’intégration actuelle ne valide pas le RHO.

Les trois paliers stricts 5, 30 et 100 RHO sont verts :

- [5 RHO — run 30565853248](https://github.com/mickaelbegon/cocofest/actions/runs/30565853248);
- [30 RHO — run 30570144903](https://github.com/mickaelbegon/cocofest/actions/runs/30570144903);
- [100 RHO — run 30573284484](https://github.com/mickaelbegon/cocofest/actions/runs/30573284484).

Sur 30 RHO, les résultats réduits sont très cohérents :

| Solveur reduced | Préfixe strict | Médiane chaude | Mur-à-mur | Coût fatigue |
|---|---:|---:|---:|---:|
| IPOPT compilé | 30/30 | 0.726 s | 154.5 s | 256.519 |
| MadNLP/MUMPS | 30/30 | 0.847 s | 68.0 s | 256.415 |
| FATROP compilé | 30/30 | 1.420 s | 197.8 s | 256.488 |

L’écart de coût entre les trois solveurs reduced est seulement `0.041 %`.
Les patrons de stimulation sont également presque identiques aux cycles 10 et
30 : les RMSE par muscle restent sous `0.163 µs` au cycle 10 et `0.103 µs`
au cycle 30 par rapport à IPOPT. Cela valide bien la reproductibilité
numérique de la formulation reduced.

Sur 100 RHO, cette concordance est conservée :

| Solveur reduced | Préfixe strict | Médiane chaude | Mur-à-mur | Coût fatigue | AUC | Capacité minimale |
|---|---:|---:|---:|---:|---:|---:|
| IPOPT compilé | 100/100 | **0.761 s** | 229.1 s | 4343.502 | 9.301 | 0.900366 |
| MadNLP/MUMPS | 100/100 | 0.889 s | **200.4 s** | 4343.271 | 9.297 | 0.900351 |
| FATROP compilé | 100/100 | 1.284 s | 247.5 s | 4343.535 | 9.299 | 0.900361 |

L’étendue du coût n’est plus que `0.0061 %`. IPOPT a le meilleur temps chaud,
tandis que MadNLP a le meilleur temps mur-à-mur sur cette machine. La fatigue
est principalement portée par le Biceps, dont la capacité finale vaut environ
`0.9004`; le coût instantané continue à augmenter, mais aucun des trois
solveurs reduced n’échoue par fatigue avant le cycle 100.

Depuis cette campagne Radau 3, le
[gate scientifique 30754413003](https://github.com/mickaelbegon/cocofest/actions/runs/30754413003)
a réintégré cinq RHO avec DOP853 (`rtol=1e-11`, `atol=1e-13`). Le drift
normalisé reste autour de `4.2–4.4 %` en Radau 4 et `1.95–2.17 %` en Radau 5,
mais tombe à `0.094 %` avec IPOPT/Radau 6 et `0.336 %` avec
MadNLP/Radau 6. MadNLP/Radau 6 reduced prend `30.836 s` solveur pour les cinq
RHO, contre `62.493 s` pour IPOPT, et donne un coût DOP853 de `19.198219`.
Je recommande donc provisoirement MadNLP/MUMPS reduced Radau 6 pour la cible
scientifique, sans annoncer encore une certification avant le palier 30.

Il faut toutefois être très prudent avec le full. IPOPT résout 28 fenêtres sur
30 et MadNLP 26 sur 30 lorsqu’elles sont regardées isolément, mais tous deux
échouent au RHO 2. Leur préfixe strict reste donc limité à un cycle. ACADOS
full retourne un statut NLP valide au premier RHO, mais l’audit détecte un
résidu de vitesse tangentielle de `0.804 rad/s`; son préfixe physique est nul.
L'ancien échec FATROP full ne venait pas du modèle : Bioptim attachait au
stage zéro les 120 lignes de la contrainte de cadence évaluée en multi-thread.
Le nouveau SHA les redistribue par stage. Le vrai OCP full passe localement
`1/1` (124 itérations, `14.344 s` solveur, audit physique valide); il faut
encore le confirmer sur le prochain gate Linux avant de comparer son endurance.

À 100 RHO, IPOPT et MadNLP résolvent respectivement 98 et 94 fenêtres
isolées, mais leur préfixe strict reste toujours d’un seul cycle. Les solutions
obtenues après le premier échec ne forment donc ni une trajectoire d’endurance
continue, ni une comparaison full/reduced certifiée.

La faible fatigue reduced par rapport aux anciens résultats full ne doit donc
pas être interprétée comme un gain physiologique démontré. Sur le seul premier
RHO comparable, l’objectif reduced est environ `0.936 %` plus faible que le
full avec IPOPT comme avec MadNLP. C’est un petit écart, mais nous n’avons pas
encore un préfixe full assez long pour comparer la fatigue cumulée.

ACADOS reduced est très rapide, environ `0.1 s` pour le premier RHO, mais
échoue au transfert vers le RHO 2 avec `ACADOS_MINSTEP`. `FIXED_STEP`,
Anderson, RTI, IRK léger et la phase I donnent actuellement le même préfixe
d’un cycle. Une piste importante est la gestion du warm start des PW : le
changement maximal entre les cycles 1 et 2 atteint environ `469 µs` au Biceps
et `80 µs` au Triceps, alors que le trust region ACADOS courant ne permet que
`10 µs`. Comme les P95 restent sous `0.3 µs`, il faut probablement relâcher
seulement les rares nœuds qui changent de branche active, pas élargir toutes
les bornes.

Pour reproduire un palier sur le calculateur, avec `N=5`, puis `30`, puis
`100` :

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

Les artefacts contiennent les temps mur-à-mur et par RHO, la convergence NLP,
le préfixe physique, les coutures d’état, le coût et la fatigue des quatre
muscles, les patrons aux cycles 10 et 30 et les variations de PW. Le point
principal à vérifier sur le calculateur est la robustesse full au RHO 2. Pour
ACADOS reduced, il faut ensuite tester un warm start de PW sensible aux
changements de branche active, puis une restauration de faisabilité suivie
d’un OCP d’optimalité, tous deux précompilés. Enfin, il faut remesurer le gain
mur-à-mur des solveurs reduced après amortissement des compilations et caches.
