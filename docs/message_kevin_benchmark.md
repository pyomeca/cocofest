# Message proposé pour Kevin

Salut Kevin,

J’ai adapté et durci le benchmark du MHE de pédalage pour comparer IPOPT,
MadNLP et Alpaqa sur exactement le même problème assisté. Le couple externe
est une **assistance de 0,20 N·m** : comme le pédalier tourne avec
`qdot < 0`, le couple généralisé constant vaut `-0.20 N.m`, soit environ
`+1.2566 W` de puissance mécanique à la cadence nominale. L’objectif est
uniquement la minimisation de la fatigue.

Le NLP commun utilise :

- la dynamique `periodic_node` ;
- une collocation directe Radau de degré 3 ;
- un horizon MHE d’un cycle par RHO, avec 30 stimulations par cycle ;
- le scaling complet des états et `0.002 rad` de tolérance sur l’angle
  terminal du pédalier ;
- 30 RHO demandés et deux échecs physiques consécutifs avant l’arrêt.

Le workflow prépare d’abord un seed IPOPT sur le **problème assisté cible** et
le certifie physiquement. Le vieux seed à `+0.22 N.m` résistif n’est utilisé
que comme trajectoire primale de continuation vers `-0.20 N.m`; il n’est
jamais présenté comme une solution assistée. Les trois jobs téléchargent
ensuite exactement le même artifact immuable.

MadNLP et Alpaqa reçoivent un hot start primal complet : états et contrôles
sont décalés d’un cycle, extrapolés, projetés dans leurs bornes et les états de
fatigue restent continus. Un raffinement IPOPT périodique certifié initialise
leur premier RHO. MadNLP ne réutilise pas les multiplicateurs, car le runtime
épinglé ne supporte pas proprement cette option. IPOPT réutilise les
multiplicateurs de bornes.

Le criblage d’options est ici :

<https://github.com/mickaelbegon/cocofest/actions/runs/30292129183>

Il montre que MadNLP doit être resserré à `tol=1e-8` : à `1e-6`, le quatrième
RHO dépasse le seuil physique commun, alors qu’à `1e-8` les quatre RHO du
criblage restent sous `1.69e-8`. Le solveur linéaire par défaut de ce runtime
est bien MUMPS; le préciser explicitement donne la même trajectoire. UMFPACK
est environ 64 % plus lent. Pour Alpaqa, laisser la pénalité initiale en mode
automatique améliore le résidu d’environ un facteur 71 par rapport au défaut
testé, sans encore converger en 60 s.

Le run Linux final de référence avec ces choix est :

<https://github.com/mickaelbegon/cocofest/actions/runs/30297904541>

Sur le runner GitHub, `nproc` expose 4 cœurs et le benchmark passe donc
`n_threads=4` à Bioptim/CasADi. Les pools imbriqués OpenMP, BLAS, MKL,
NumExpr et Julia restent à un thread pour éviter la sur-souscription. Cela
utilise tous les cœurs pour l’évaluation du graphe, pas quatre factorisations
linéaires parallèles.

Résultats du run final :

| Solveur | Préfixe validé | RHO tentés | Préparation | Somme des RHO tentés | Médiane chaude | P90 chaud | Mur-à-mur |
|---|---:|---:|---:|---:|---:|---:|---:|
| IPOPT-MUMPS (`tol=1e-6`) | 30/30 | 30 | 23.58 s | 204.92 s | 6.232 s | 8.243 s | 236.40 s |
| MadNLP-MUMPS (`tol=1e-8`) | 30/30 | 30 | 56.60 s | 188.40 s | 5.853 s | 7.806 s | 252.69 s |
| Alpaqa (`tol=1e-6`) | 0/30 | 2 | 58.76 s | 1200.03 s | — | — | 1260.25 s |

IPOPT converge et passe la vérification indépendante des bornes sur les
30 RHO. Sa capacité musculaire minimale reste élevée
(`min(A/A_scale)=0.98334`) : **30 cycles caractérisent les performances et les
patrons, mais ne provoquent pas encore un échec par fatigue**. Il faudra un run
d’endurance de plusieurs centaines, probablement proche de 1000 RHO, pour
répondre à cette autre question.

MadNLP valide maintenant les 30 RHO avec une infaisabilité maximale de
`2.57e-7`, contre `9.64e-7` pour IPOPT. Cette meilleure précision vient aussi
d’une tolérance interne 100 fois plus stricte. Sur ce run unique, MadNLP gagne
environ 6 % sur la médiane chaude, 5 % sur le P90 et 8 % sur la somme des RHO.
Il perd toutefois au mur-à-mur, car son raffinement IPOPT périodique ajoute
environ 33 s de préparation. Au gain moyen observé, ce surcoût ne serait amorti
qu’autour de 60 RHO. Il faut au moins trois répétitions appariées sur une
machine contrôlée avant de traiter ce petit gain comme réel.

Alpaqa atteint deux fois `SOLVER_RET_LIMITED` après 600 s. Le premier candidat
est intéressant pour le diagnostic : il est physiquement faisable
(`3.51e-6 < 1e-5`) et son objectif est proche d’IPOPT, mais le solveur n’a pas
certifié la convergence/stationnarité. Il n’est donc pas compté comme RHO
validé. Le second essai, alimenté uniquement par le primal décalé et projeté,
se dégrade à `4.57e-2` d’infaisabilité. Les deux fenêtres réalisent environ
48 000 évaluations de `psi` chacune et consomment près de 3.5 cœurs sur les
quatre disponibles. En l’état, Alpaqa n’est pas adapté à ce MHE de collocation.

Les patrons IPOPT et MadNLP sont presque identiques au RHO 10 :

| Muscle | IPOPT moyenne | MadNLP moyenne | RMSE réalignée sur l’angle |
|---|---:|---:|---:|
| Biceps | 133.790 µs | 133.835 µs | 0.227 µs |
| Deltoïde antérieur | 131.408 µs | 131.405 µs | 0.009 µs |
| Deltoïde postérieur | 131.408 µs | 131.405 µs | 0.009 µs |
| Triceps | 173.699 µs | 173.550 µs | 0.759 µs |

Au RHO 30, ils trouvent en revanche deux répartitions biceps/triceps très
différentes :

| Muscle | IPOPT moyenne | MadNLP moyenne | RMSE réalignée sur l’angle |
|---|---:|---:|---:|
| Biceps | 156.101 µs | 134.303 µs | 100.088 µs |
| Deltoïde antérieur | 131.407 µs | 131.405 µs | 0.006 µs |
| Deltoïde postérieur | 131.409 µs | 131.405 µs | 0.014 µs |
| Triceps | 136.451 µs | 174.827 µs | 149.049 µs |

Le réalignement interpole les stimulations selon l’angle réel du pédalier; la
divergence ne vient donc pas seulement du déphasage intra-cycle de `0.163 rad`.
Pourtant, la fatigue exécutée ne diffère que de 0.24 %
(`162.720` contre `163.110`). Sans régularisation de la commande ni de la
cadence, le NLP admet vraisemblablement plusieurs minima locaux avec des
partages musculaires différents. Ces patrons ne sont pas physiologiquement
interchangeables sans examiner aussi `q`, `qdot`, les couples, forces et états
de fatigue. Une petite régularisation, des échanges de seeds IPOPT↔MadNLP et
des perturbations du seed sont les prochaines validations pertinentes.

Pour relancer exactement l’expérience :

```bash
gh workflow run cycling_solver_benchmark_linux.yml \
  --repo mickaelbegon/cocofest \
  --ref codex/acados-pr-refresh \
  -f runner_label=ubuntu-22.04 \
  -f cycles=30 \
  -f crank_assistance_nm=0.20 \
  -f terminal_wheel_q_slack=0.002 \
  -f solver_max_iterations=2000
```

L’artifact final `cycling-fatigue-kevin-report-*` contient le rapport Markdown,
le JSON combiné, `rho-timings.csv`, `stimulation-patterns.csv` et les logs.

Points sensibles à garder en tête :

- MadNLP passe ici le seuil strict sur 30/30 seulement après avoir séparé le
  seuil physique (`1e-5`) de sa tolérance interne et resserré celle-ci à
  `1e-8`. Le résultat est convaincant, mais n’est encore qu’une répétition
  longue sur un runner partagé.
- Alpaqa est construit avec le fork de compatibilité déclaré par CasADi 3.7.2,
  pas avec l’Alpaqa amont moderne. Les conclusions ne devront pas être
  généralisées à la version actuelle d’Alpaqa. Son premier candidat est
  faisable, mais l’absence de statut de convergence et l’échec du RHO suivant
  l’excluent du MHE actuel.
- IPOPT, MadNLP et Alpaqa utilisent deux commits Bioptim d’intégration
  distincts. Le rapport les enregistre; le NLP est comparé automatiquement,
  mais cette provenance doit rester visible.
- IPOPT utilise MUMPS dans l’action publique. MA57 nécessite CoinHSL sous
  licence et doit être évalué sur un runner privé.
- La compilation du NLP n’est pas activée : sur Apple Silicon, elle augmentait
  le temps chaud et surtout le coût de préparation. Alpaqa ne supporte pas ce
  chemin avec le plugin CasADi utilisé ici.

En l’état, IPOPT-MUMPS reste la référence la plus simple et la meilleure à
30 RHO en mur-à-mur. MadNLP-MUMPS devient une alternative crédible et robuste
sur ce test : il est légèrement plus rapide une fois chaud, mais son coût de
préparation doit être amorti et la divergence tardive des patrons doit être
étudiée. Alpaqa est fonctionnel et utilise bien les cœurs disponibles, mais il
n’est pas pertinent pour ce NLP de collocation sans travail supplémentaire
sur ALM/PANOC, le scaling et surtout une formulation moins redondante.
