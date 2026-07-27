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

<https://github.com/mickaelbegon/cocofest/actions/runs/30304318862>

Les deux jobs ont utilisé les quatre cœurs exposés par le runner pour
l’évaluation CasADi/Bioptim. Les pools OpenMP, BLAS, MKL, NumExpr et Julia
imbriqués restent à un thread pour éviter la sur-souscription.

## Résultats à 100 RHO déjà établis

| Solveur | RHO résolus | Préfixe strict | Préparation | Somme des RHO | Médiane chaude | P90 chaud | Mur-à-mur |
|---|---:|---:|---:|---:|---:|---:|---:|
| IPOPT-MUMPS (`tol=1e-6`) | 100/100 | 100/100 | 23.03 s | 818.74 s | 8.241 s | 11.130 s | 866.22 s |
| MadNLP-MUMPS (`tol=1e-8`) | 99/100 | 85/100 | 45.72 s | 815.33 s | 5.623 s | 7.745 s | 881.48 s |

Il faut distinguer les deux colonnes MadNLP. Le **préfixe strict** s’arrête au
premier échec afin de ne pas considérer comme exécutée une trajectoire passant
par un RHO non certifié. MadNLP n’a toutefois pas cessé de résoudre : son
RHO 86 atteint `SOLVER_RET_LIMITED` après 2000 itérations et 187.12 s, avec
`4.85e-2` d’infaisabilité, puis les RHO 87 à 100 convergent et sont
physiquement faisables. Il y a donc 99 solutions certifiées sur 100 et jamais
deux échecs consécutifs.

IPOPT résout le même RHO 86 en 76 itérations, 8.59 s et avec une infaisabilité
de `7.34e-7`. L’échec MadNLP ressemble donc davantage à un accident de
robustesse locale ou de hot start qu’à une impossibilité physique créée par la
fatigue. Comme les 14 fenêtres suivantes récupèrent, on ne peut pas non plus
présenter le cycle 86 comme le point de défaillance physiologique.

Hors cet unique outlier, MadNLP est nettement plus rapide une fois chaud. Sur
les 85 premières fenêtres communes, son temps par RHO est environ 18 % plus
faible en médiane. Sur les RHO 87–100, sa médiane est 7.16 s contre 8.59 s
pour IPOPT. Mais l’outlier de 187 s et les 22.69 s supplémentaires de
préparation annulent entièrement cet avantage : IPOPT gagne finalement
15.26 s au mur-à-mur. Pour un MHE en production, cette variance extrême compte
plus que la bonne médiane.

IPOPT reste physiquement certifié sur les 100 fenêtres, avec une infaisabilité
maximale de `9.64e-7`. Sa capacité musculaire minimale à la fin vaut encore
`A/A_scale=0.92495` pour le biceps. Le benchmark de 100 cycles ne va donc pas
encore jusqu’à l’échec par fatigue ; il caractérise surtout la robustesse et le
coût des solveurs.

Les patrons IPOPT et MadNLP sont presque identiques au RHO 10. Au RHO 30, ils
adoptent en revanche des partages biceps/triceps très différents malgré des
coûts de fatigue voisins. L’écart persiste après réalignement sur l’angle réel
du pédalier, ce qui indique plusieurs minima locaux plausibles dans un problème
sans régularisation des commandes. Le patron du RHO 100 est exporté pour IPOPT
(biceps moyen `154.1 µs`, triceps `148.8 µs`), mais volontairement indisponible
pour MadNLP, car il se trouve après la rupture du préfixe strict.

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

En synthèse, **IPOPT-MUMPS reste la référence recommandée** : il est le seul à
valider sans interruption les 100 RHO et il gagne aussi au mur-à-mur.
**MadNLP-MUMPS est prometteur mais pas encore robuste** : son régime normal est
plus rapide, son hot start fonctionne jusqu’à un horizon significatif et il
récupère après l’échec, mais une seule fenêtre pathologique suffit à effacer
le gain. **Alpaqa est laissé de côté** pour cette formulation de collocation,
car aucun réglage testé n’a fourni une continuation MHE convergente.

**Fatrop est maintenant inclus dans le prochain lancement 100 RHO**, mais je
ne le classe pas encore face aux deux résultats ci-dessus. Son résultat local
court valide l’intégration et la faisabilité, pas sa robustesse à la fatigue.
Le rapport GitHub signalera aussi explicitement que son ordre et son scaling
numériques diffèrent, afin que son temps ne soit pas présenté comme un simple
changement de backend à NLP numérique strictement identique.
