# Message proposé pour Kevin

Salut Kevin,

J’ai adapté et durci le benchmark du MHE de pédalage pour comparer IPOPT,
Fatrop et MadNLP sur le même problème assisté. Alpaqa est désormais retiré de la matrice
d’endurance : le meilleur réglage testé n’a validé aucun RHO, a atteint deux
limites de 600 s et a fini le second RHO à `4.57e-2` d’infaisabilité. Son
intégration et ses résultats historiques restent documentés pour la
reproductibilité, mais il ne consomme plus de temps dans le benchmark normal.

Le couple externe est une **assistance de 0,20 N·m**. Comme le pédalier tourne
avec `qdot < 0`, le couple généralisé constant vaut `-0.20 N.m`, soit environ
`+1.26 W` de puissance mécanique à la cadence nominale. L’objectif du NLP est
uniquement la minimisation de la fatigue.

Le problème commun utilise :

- la dynamique `periodic_node` ;
- une collocation directe Radau de degré 3 ;
- un horizon MHE d’un cycle par RHO, avec 30 stimulations par cycle ;
- le scaling complet des états pour IPOPT et MadNLP ;
- une tolérance de `0.002 rad` sur la progression angulaire terminale ;
- 100 RHO demandés et un arrêt après deux échecs consécutifs.

Le workflow construit d’abord un seed IPOPT sur le problème assisté cible et
le certifie physiquement. IPOPT, Fatrop et MadNLP téléchargent ensuite le même artifact
immuable et utilisent le même commit Bioptim. Le vieux seed résistif n’est
qu’une trajectoire de continuation et n’est jamais présenté comme solution du
problème assisté.

MadNLP reçoit un hot start primal complet : états et contrôles sont décalés
d’un cycle, extrapolés puis projetés dans leurs bornes, avec continuité des
états de fatigue. Un raffinement IPOPT périodique certifié prépare son premier
RHO. La réutilisation des multiplicateurs MadNLP reste désactivée, car le
runtime épinglé ne la supporte pas proprement. IPOPT réutilise les
multiplicateurs de bornes.

Fatrop reçoit le même hot start primal et le même raffinement IPOPT initial,
mais impose deux adaptations numériques visibles dans le rapport. Il utilise
un ordre temporel des variables (`time_major`) et aucun scaling des états :
avec le scaling complet, sa détection automatique refuse les gaps de
collocation. Il applique aussi une relaxation relative interne des bornes. Sur
les états de capacité proches de 7000, cela autorisait environ `7e-5` de
dépassement malgré une convergence native. L’interface Bioptim épinglée
resserre donc uniquement les bornes envoyées à Fatrop de `1e-8`, puis Cocofest
audite la solution contre les bornes physiques originales. On ne desserre pas
le seuil commun.

Le smoke test local de cinq RHO valide les cinq fenêtres Fatrop : violation
de contrainte maximale `3.25e-9`, aucune violation de borne physique, 81 à 88
itérations et `4.80--5.93 s` par résolution (`5.24 s` de médiane). C’est
prometteur, mais insuffisant pour conclure sur 100 cycles.

Le criblage d’options est ici :

<https://github.com/mickaelbegon/cocofest/actions/runs/30292129183>

Il a retenu MadNLP-MUMPS avec `tol=1e-8`. À `1e-6`, la quatrième fenêtre
dépasse le seuil physique commun de `1e-5`; à `1e-8`, les quatre fenêtres du
criblage restent sous `1.69e-8`. MUMPS est bien le solveur linéaire par défaut
de ce runtime, tandis qu’UMFPACK a été environ 64 % plus lent.

Le benchmark Linux 100 RHO est ici :

<https://github.com/mickaelbegon/cocofest/actions/runs/30309452077>

Les trois jobs ont utilisé les quatre cœurs exposés par le runner pour
l’évaluation CasADi/Bioptim. Les pools OpenMP, BLAS, MKL, NumExpr et Julia
imbriqués restent à un thread pour éviter la sur-souscription.

## Résultats à 100 RHO

| Solveur | RHO résolus | Préfixe strict | Préparation | Somme des RHO | Médiane chaude | P90 chaud | Mur-à-mur |
|---|---:|---:|---:|---:|---:|---:|---:|
| IPOPT-MUMPS (`tol=1e-6`) | 100/100 | 100/100 | 23.82 s | 830.20 s | 8.311 s | 11.190 s | 878.90 s |
| Fatrop (`tol=1e-6`) | 100/100 | 100/100 | 42.96 s | 1148.17 s | 11.979 s | 14.016 s | 1209.02 s |
| MadNLP-MUMPS (`tol=1e-8`) | 100/100 | 100/100 | 54.90 s | 799.46 s | 6.252 s | 9.278 s | 877.35 s |

Les trois solveurs convergent et passent l’audit physique sur les 100 fenêtres.
Les infaisabilités maximales sont `9.64e-7` pour IPOPT, `9.55e-7` pour
Fatrop et `1.00e-6` pour MadNLP, donc nettement sous le seuil commun `1e-5`.
Fatrop ne viole aucune borne physique après compensation de sa relaxation
interne.

Fatrop est régulier dans ce run, sans échec ni outlier extrême, mais il est ici
nettement le plus lent : +44 % sur la médiane chaude
et +38 % au mur-à-mur par rapport à IPOPT. Son coût augmente avec l’horizon :
77 itérations et 7.60 s de médiane sur les RHO 1–10, contre 122 itérations et
13.33 s sur les RHO 91–100. Il faut toutefois rester prudent dans
l’interprétation : sa détection structurée impose actuellement l’absence de
scaling des états, alors qu’IPOPT et MadNLP utilisent le scaling complet. Le
rapport signale explicitement cet unique écart de configuration.

MadNLP a la meilleure médiane et la plus petite somme des RHO. Il subit
toutefois un outlier au RHO 99 : 969 itérations et 99.61 s. Cette fois il
converge, contrairement à l’échec du RHO 86 observé dans le run précédent.
Son supplément de préparation annule presque exactement son avantage en
régime chaud : 877.35 s contre 878.90 s pour IPOPT au mur-à-mur. Cette
variabilité confirme qu’une répétition unique ne suffit pas pour le classer
comme plus robuste qu’IPOPT.

IPOPT et Fatrop suivent des minima locaux proches : Fatrop finit à
`min(A/A_scale)=0.92453` contre `0.92495` pour IPOPT, et leurs patrons restent
très corrélés. Au RHO 10, les RMSE biceps/triceps par rapport à IPOPT valent
`0.10/0.49 µs`; au RHO 30, `1.84/5.21 µs`. MadNLP trouve une solution beaucoup
moins fatiguante (`min(A/A_scale)=0.94777`, objectif exécuté environ 34 % plus
bas), avec un partage biceps/triceps radicalement différent au RHO 30
(`100/140 µs` de RMSE, corrélations proches de zéro). Sans régularisation des
commandes, cela indique plusieurs bassins de solutions ; il faut valider les
forces, vitesses et critères physiologiques avant d’interpréter ce minimum
comme réellement supérieur.

Le benchmark de 100 cycles ne va toujours pas jusqu’à l’échec par fatigue :
même la trajectoire IPOPT la plus fatiguée conserve environ 92.5 % de la
capacité minimale. Il caractérise ici la robustesse et le coût des solveurs,
pas le cycle physiologique de défaillance.

Un autre point sensible apparaît seulement sur l’horizon long : la tolérance
terminale de `0.002 rad` est respectée à chaque RHO, mais elle est presque
toujours utilisée dans le même sens à partir d’environ 30 cycles. L’erreur
angulaire accumulée d’IPOPT atteint `0.1587 rad` à 100 cycles, soit environ
`9.1°`. Le solveur converge, mais cette dérive rend l’état terminal moins
précis qu’on le souhaite pour une étude d’endurance. Le prochain essai
scientifique devrait comparer `0`, `1e-4` et `0.002 rad`, sans modifier
rétroactivement ce benchmark de performance.

Pour relancer exactement le benchmark :

```bash
gh workflow run cycling_solver_benchmark_linux.yml \
  --repo mickaelbegon/cocofest \
  --ref codex/acados-pr-refresh \
  -f runner_label=ubuntu-22.04 \
  -f cycles=100 \
  -f crank_assistance_nm=0.20 \
  -f terminal_wheel_q_slack=0.002 \
  -f solver_max_iterations=2000
```

L’artifact final `cycling-fatigue-kevin-report-*` contient le rapport Markdown,
le JSON combiné, `rho-timings.csv`, `stimulation-patterns.csv` et tous les
logs. Le rapport distingue maintenant « RHO résolus » et « préfixe strict »
pour rendre une récupération après échec visible.

En synthèse, **IPOPT-MUMPS reste la référence recommandée** : robuste, plus
rapide que Fatrop et moins variable que MadNLP. **MadNLP-MUMPS reste le
candidat le plus performant une fois chaud**, mais son outlier de 100 s et les
résultats différents entre répétitions imposent encore de travailler le hot
start et de répéter les mesures. **Fatrop est désormais pleinement
fonctionnel et robuste sur 100 RHO**, mais il n’apporte pas de gain de temps
avec la compatibilité actuelle sans scaling ; son intérêt principal est une
seconde méthode structurée indépendante pour valider les solutions.
**Alpaqa reste laissé de côté**, car aucun réglage testé n’a fourni une
continuation MHE convergente.
