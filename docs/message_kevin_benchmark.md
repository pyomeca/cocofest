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

Le run Linux final de référence est :

<https://github.com/mickaelbegon/cocofest/actions/runs/30287669771>

Sur le runner GitHub, `nproc` expose 4 cœurs et le benchmark passe donc
`n_threads=4` à Bioptim/CasADi. Les pools imbriqués OpenMP, BLAS, MKL,
NumExpr et Julia restent à un thread pour éviter la sur-souscription. Cela
utilise tous les cœurs pour l’évaluation du graphe, pas quatre factorisations
linéaires parallèles.

Résultats du run final :

| Solveur | Préfixe validé | RHO tentés | Préparation | Somme des RHO tentés | Médiane chaude | P90 chaud | Mur-à-mur |
|---|---:|---:|---:|---:|---:|---:|---:|
| IPOPT-MUMPS | 30/30 | 30 | 23.40 s | 202.31 s | 6.130 s | 8.116 s | 233.53 s |
| MadNLP | 3/30 | 11 | 53.69 s | 61.43 s | 5.353 s | 5.472 s | 118.38 s |
| Alpaqa | 0/30 | 2 | 52.96 s | 1200.04 s | — | — | 1254.47 s |

IPOPT converge et passe la vérification indépendante des bornes sur les
30 RHO. Sa capacité musculaire minimale reste élevée
(`min(A/A_scale)=0.98334`) : **30 cycles caractérisent les performances et les
patrons, mais ne provoquent pas encore un échec par fatigue**. Il faudra un run
d’endurance de plusieurs centaines, probablement proche de 1000 RHO, pour
répondre à cette autre question.

MadNLP retourne un statut natif de succès pour ses 11 RHO tentés, mais la
reconstruction indépendante du NLP détecte une infaisabilité de
`1.2766e-5` au RHO 4, au-dessus du seuil scientifique `1e-5`. Des RHO suivants
repassent ponctuellement sous le seuil, mais le préfixe comparable reste
strictement limité aux trois premiers. Les RHO 10 et 11 échouent ensuite
consécutivement (`1.0463e-5` et `1.1523e-5`), ce qui déclenche l’arrêt demandé
après deux chances. Son temps mur-à-mur plus court n’est pas comparable,
puisqu’il n’a tenté que 11 RHO et n’en valide que 3.

Les secondes brutes des runners GitHub ont une variabilité importante. Sur
trois runs corrigés, les médianes chaudes observées vont de 3.83 à 6.13 s pour
IPOPT et de 5.32 à 5.82 s pour MadNLP. La médiane de ces trois mesures est
4.43 s pour IPOPT et 5.35 s pour MadNLP, mais le dernier run isolé inverse le
classement. Il est donc raisonnable de conclure qu’IPOPT et MadNLP sont du même
ordre de grandeur sur ce runner, pas d’annoncer un facteur d’accélération
précis. La robustesse reste en revanche sans ambiguïté : 30/30 contre 3/30.

Alpaqa atteint `SOLVER_RET_LIMITED` après 600 s sur chacun de ses deux RHO et
n’en valide aucun. Les infaisabilités finales sont `8.7335e-5` puis
`6.3190e-3`. Le CasADi compilé pour ce job est bien multithreadé : environ
3.5 cœurs sur 4 sont effectivement occupés pendant les évaluations, et le
premier RHO effectue environ 69 000 évaluations de `psi` au lieu d’environ
40 000 dans le run sériel antérieur. Cela améliore le premier résidu d’un
facteur proche de 2.8, mais ne suffit pas à converger.

Le second essai Alpaqa utilise uniquement la trajectoire primale décalée et
projetée; les multiplicateurs du premier essai limité sont explicitement
effacés. Sa dégradation montre que le problème ne venait pas seulement du hot
start dual. Il faut désormais traiter le réglage ALM/PANOC, le scaling des
contraintes et probablement la réduction de dimension du NLP avant de
consacrer un autre run de 30 RHO à ce backend.

Le benchmark exporte les patrons seulement si le cycle appartient au préfixe
validé. Les patrons MadNLP aux RHO 10 et 30 sont donc volontairement absents,
et non perdus; il en va de même pour Alpaqa, qui n’a aucun cycle validé. Pour
IPOPT :

| RHO | Muscle | Largeur moyenne | Maximum | Près de la borne basse | Près de la borne haute |
|---:|---|---:|---:|---:|---:|
| 10 | Biceps | 133.790 µs | 202.612 µs | 96.7 % | 0 % |
| 10 | Deltoïde antérieur | 131.408 µs | 131.453 µs | 100 % | 0 % |
| 10 | Deltoïde postérieur | 131.408 µs | 131.451 µs | 100 % | 0 % |
| 10 | Triceps | 173.699 µs | 599.940 µs | 90.0 % | 6.7 % |
| 30 | Biceps | 156.101 µs | 599.879 µs | 93.3 % | 3.3 % |
| 30 | Deltoïde antérieur | 131.407 µs | 131.435 µs | 100 % | 0 % |
| 30 | Deltoïde postérieur | 131.409 µs | 131.478 µs | 100 % | 0 % |
| 30 | Triceps | 136.451 µs | 268.812 µs | 93.3 % | 0 % |

Ces patrons ne doivent pas être interprétés seuls. Il n’y a pas de
régularisation des contrôles ni de cadence intra-cycle imposée; l’assistance
peut aussi demander aux muscles de réguler ou freiner. Le CSV fournit donc
également la phase et la vitesse réelles du pédalier.

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

- MadNLP semble converger selon son propre statut, mais sa précision physique
  est marginalement insuffisante avec notre seuil strict. Il faut donc éviter
  de conclure à partir du seul `SOLVER_RET_SUCCESS`.
- Alpaqa est construit avec le fork de compatibilité déclaré par CasADi 3.7.2,
  pas avec l’Alpaqa amont moderne. Les conclusions ne devront pas être
  généralisées à la version actuelle d’Alpaqa. Dans la configuration testée,
  il est beaucoup trop lent et ne converge pas.
- IPOPT, MadNLP et Alpaqa utilisent deux commits Bioptim d’intégration
  distincts. Le rapport les enregistre; le NLP est comparé automatiquement,
  mais cette provenance doit rester visible.
- IPOPT utilise MUMPS dans l’action publique. MA57 nécessite CoinHSL sous
  licence et doit être évalué sur un runner privé.
- La compilation du NLP n’est pas activée : sur Apple Silicon, elle augmentait
  le temps chaud et surtout le coût de préparation. Alpaqa ne supporte pas ce
  chemin avec le plugin CasADi utilisé ici.

En l’état, IPOPT-MUMPS reste clairement la référence robuste. MadNLP exécute le
problème et bénéficie bien du hot start primal, mais il n’est pas encore assez
fiable à `1e-5` pour le MHE d’endurance. Alpaqa est fonctionnel et utilise bien
les cœurs disponibles, mais il n’est pas pertinent pour ce NLP sans une phase
de recherche supplémentaire sur ALM/PANOC et sur la formulation.
