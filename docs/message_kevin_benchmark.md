# Message proposé pour Kevin

Salut Kevin,

J’ai adapté et durci le benchmark du MHE de pédalage pour comparer IPOPT,
Fatrop et MadNLP sur le même problème assisté. L’objectif du NLP est uniquement
la minimisation de la fatigue.

Le couple externe est une **assistance de 0,20 N·m**. Comme le pédalier tourne
avec `qdot < 0`, le couple généralisé constant vaut `-0.20 N.m`, soit environ
`+1.26 W` de puissance mécanique à la cadence nominale.

Le problème commun utilise :

- la dynamique musculaire `periodic_node` ;
- une collocation directe Radau de degré 3 ;
- un horizon MHE d’un cycle par RHO, avec 30 stimulations par cycle ;
- 100 RHO demandés et un arrêt après deux échecs consécutifs ;
- une cible angulaire absolue
  `q_initial + k*(-2*pi)`, avec une bande terminale de `0.002 rad` ;
- le scaling complet des états pour IPOPT et MadNLP.

Fatrop utilise actuellement un ordre temporel des variables et aucun scaling
des états. Avec le scaling complet, sa détection structurée refuse les gaps de
collocation. Le rapport marque explicitement cet écart : ses temps ne
constituent donc pas encore une comparaison purement « backend contre
backend ».

Le workflow construit d’abord un seed IPOPT sur le problème assisté cible et
le certifie physiquement. Les trois solveurs téléchargent ensuite exactement
le même artefact et le même commit Bioptim. Le vieux seed résistif n’est
utilisé que comme point de départ d’une continuation vers le problème
assisté ; il n’est jamais présenté comme une solution du nouveau problème.

IPOPT réutilise les multiplicateurs de bornes. Fatrop et MadNLP reçoivent un
hot start primal complet : états et contrôles sont décalés d’un cycle,
extrapolés puis projetés dans leurs bornes, avec continuité des états de
fatigue. Un raffinement IPOPT périodique certifié prépare leur premier RHO.
La réutilisation des multiplicateurs MadNLP reste désactivée, car le runtime
épinglé ne la supporte pas proprement.

MadNLP utilise ici `PardisoMKLSolver` via le commit libMad
`5529f23a6bff33c566ad954da38d352f1f172356`. Le workflow vérifie le runtime
PARDISO, construit le pont CasADi compatible avec la nouvelle ABI `libMad.so`,
puis exécute un smoke test via le véritable adaptateur Cocofest avant l’OCP.
Cette voie est Linux x86-64 uniquement ; elle n’est pas disponible nativement
sur les Mac Apple Silicon.

Le benchmark complet est ici :

<https://github.com/mickaelbegon/cocofest/actions/runs/30363688991>

Les runners exposaient quatre cœurs. Les évaluations CasADi/Bioptim utilisent
ces quatre cœurs ; PARDISO reçoit également les quatre threads MKL. Les pools
OpenMP/BLAS/NumExpr/Julia imbriqués restent à un thread pour éviter la
sur-souscription.

## Résultats à 100 RHO

| Solveur | RHO validés | Préparation | Somme des RHO | Médiane chaude | P90 chaud | Mur-à-mur |
|---|---:|---:|---:|---:|---:|---:|
| IPOPT-MUMPS (`tol=1e-6`) | 100/100 | 23.76 s | 730.71 s | 7.110 s | 9.434 s | 778.99 s |
| Fatrop (`tol=1e-6`) | 100/100 | 56.86 s | 1230.43 s | 11.719 s | 17.607 s | 1312.02 s |
| MadNLP-PARDISO (`tol=1e-8`) | 100/100 | 51.60 s | 1060.00 s | 8.679 s | 10.827 s | 1135.93 s |

Les trois solveurs convergent sur les 100 fenêtres et passent l’audit physique
indépendant. Leurs infaisabilités maximales sont `9.71e-7` pour IPOPT,
`9.43e-7` pour Fatrop et `1.00e-6` pour MadNLP, sous le seuil commun `1e-5`.
Il n’y a aucun échec, aucune récupération après échec et aucun arrêt
prématuré.

IPOPT est le plus rapide sur ce run. Par rapport à IPOPT, Fatrop est environ
65 % plus lent sur la médiane chaude et 68 % au mur-à-mur. MadNLP-PARDISO est
22 % plus lent sur la médiane chaude et 46 % au mur-à-mur.

MadNLP présente surtout une queue lourde : le RHO 75 prend `141.22 s` et
1177 itérations, le RHO 82 `63.55 s` et 533 itérations, et le RHO 60
`24.61 s` et 209 itérations. Les trois convergent et restent physiquement
faisables. La médiane seule donnerait donc une image trop favorable.

Le run PARDISO précédent donne pratiquement les mêmes valeurs, ce qui rend ce
comportement reproductible sur ce problème. L’ancien benchmark
MadNLP-MUMPS était plus rapide, mais utilisait encore la cible angulaire
relative qui pouvait dériver. Il ne s’agit donc pas d’une comparaison
PARDISO/MUMPS strictement appariée. À ce stade, il n’y a **aucune preuve que
PARDISO accélère cet OCP** ; il faut exécuter MUMPS et PARDISO avec le même
code absolu, le même seed et plusieurs répétitions avant d’attribuer l’écart au
solveur linéaire.

## Fatigue et patrons de stimulation

| Solveur | Objectif de fatigue exécuté | Minimum `A/A_scale` | Fatigue moyenne max. |
|---|---:|---:|---:|
| IPOPT | 2177.52 | 0.93750 | 0.03333 |
| Fatrop | 1751.06 | 0.94215 | 0.02552 |
| MadNLP-PARDISO | 1696.11 | 0.94093 | 0.02395 |

Au cycle 10, les trois patrons sont presque identiques. Par rapport à IPOPT,
les RMSE après alignement angulaire de MadNLP valent `0.089 µs` pour le biceps
et `2.47 µs` pour le triceps ; les corrélations sont supérieures à `0.9999`.

Au cycle 30, Fatrop et MadNLP basculent ensemble dans un autre partage
biceps/triceps :

| Solveur | Biceps moyen / max. | Triceps moyen / max. |
|---|---:|---:|
| IPOPT | 158.7 / 599.7 µs | 135.5 / 252.7 µs |
| Fatrop | 134.3 / 218.3 µs | 174.1 / 600.0 µs |
| MadNLP-PARDISO | 134.4 / 221.1 µs | 173.8 / 600.0 µs |

La RMSE alignée par rapport à IPOPT atteint environ `108 µs` au biceps et
`144--145 µs` au triceps, avec des corrélations proches de zéro. Fatrop et
MadNLP restent en revanche très proches entre eux au cycle 30 (`0.49 µs` de
RMSE biceps et `1.05 µs` triceps). Les deltoïdes restent pratiquement
identiques dans les trois solutions.

Cela indique plusieurs minima locaux de partage musculaire. Comme il n’y a
pas de régularisation des commandes, une fatigue scalaire plus faible ne
suffit pas pour déclarer la solution physiologiquement meilleure. Il faut
encore comparer forces, couples musculaires, cadence et robustesse aux
perturbations ou échanger les solutions initiales entre solveurs.

Le benchmark de 100 cycles ne va pas jusqu’à l’échec par fatigue : la capacité
minimale reste entre 93,7 % et 94,2 %. Il mesure ici la robustesse de
continuation, le temps de calcul et les bassins de solution. Pour provoquer un
échec physiologique, il faudra passer à plusieurs centaines de cycles, très
probablement autour de 1000, en gardant l’arrêt après deux non-convergences
consécutives.

L’audit angulaire est maintenant réellement absolu. Il utilise l’origine fixe
du problème, corrigée du cycle de warmup consommé, et convertit le seuil de
faisabilité du vecteur NLP mis à l’échelle vers les radians. Les erreurs
absolues maximales sont `0.002005 rad` pour IPOPT, `0.002000 rad` pour Fatrop
et `0.002006 rad` pour MadNLP. Elles correspondent à la bande terminale et ne
s’accumulent pas sur 100 cycles.

Alpaqa est laissé de côté. Le meilleur réglage testé n’a validé aucun RHO, a
atteint deux limites de 600 s et a terminé le second RHO à `4.57e-2`
d’infaisabilité. Son intégration historique reste documentée pour la
reproductibilité, mais il ne fait plus partie du benchmark d’endurance.

Pour relancer exactement le benchmark :

```bash
gh workflow run cycling_solver_benchmark_linux.yml \
  --repo mickaelbegon/cocofest \
  --ref codex/acados-pr-refresh \
  -f runner_label=ubuntu-24.04 \
  -f cycles=100 \
  -f crank_assistance_nm=0.20 \
  -f terminal_wheel_q_slack=0.002 \
  -f solver_max_iterations=2000 \
  -f seed_validation_max_iterations=2000
```

L’artefact `cycling-fatigue-kevin-report-30363688991` contient le rapport
Markdown, le JSON combiné, les temps de chaque RHO, les patrons aux cycles
10, 30 et 100, ainsi que tous les logs.

En synthèse, **IPOPT-MUMPS reste la référence recommandée** : il est robuste,
le plus rapide et présente une queue de temps plus maîtrisée.
**MadNLP-PARDISO est fonctionnel et certifie 100/100 RHO**, mais il est plus
lent qu’IPOPT et subit deux outliers sévères ; son intérêt actuel est surtout
une validation indépendante et l’exploration d’un autre minimum local.
**Fatrop est également robuste sur 100/100 RHO**, mais sa compatibilité sans
scaling le rend nettement plus lent ; il est utile comme second solveur
structuré et confirme ici le bassin de faible fatigue trouvé par MadNLP.
**Alpaqa n’est pas pertinent pour ce MHE de collocation dans son état actuel.**
