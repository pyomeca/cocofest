# Accélérer la résolution des RHO de pédalage FES

Ce document décrit la meilleure méthode actuellement recommandée pour résoudre
les RHO de pédalage en minimisant la fatigue. Il s'agit d'une prescription
scientifique et numérique, pas d'un journal de développement.

L'[historique complet](development_history.md) conserve les campagnes GitHub
Actions, les variantes essayées, les échecs, les changements de SHA et les
résultats qui ont conduit aux choix ci-dessous.

Le [point de reprise](resume_and_todo.md) résume les dernières analyses et
donne une liste de tâches priorisée avec les critères permettant de les
considérer terminées.

Pour déplacer la campagne sur un nouveau calculateur, utiliser le
[prompt de continuation](continuation_prompt.md) et la
[procédure Linux 32 cœurs](linux_32core_setup.md).

## Versions reproductibles du workflow actif

La CI ne dépend pas d'un nom de branche flottant. Elle clone le fork
[`mickaelbegon/BiorbdOptim`](https://github.com/mickaelbegon/BiorbdOptim) et
effectue les checkouts par SHA complet :

- intégration multi-solveurs active :
  `dad96b90d47c36126c1e97ec35f27c499abf4b12`;
- intégration Alpaqa archivée :
  `d84e7e43534360fc048e0be26a3bd69a2abc2d77`;
- écran MadNLP/MUMPS historique, conservé uniquement pour diagnostic :
  `346eb1d445e6ba67010b96c6f16ba830185119e7`.

Le détail des anciennes révisions, des branches de provenance et des patchs
Bioptim se trouve dans
[l'historique des développements](development_history.md). Le workflow
[`cycling_solver_benchmark_linux.yml`](../../.github/workflows/cycling_solver_benchmark_linux.yml)
reste la source de vérité exécutable.

## Réponse courte

Pour obtenir aujourd'hui la meilleure combinaison de robustesse et de vitesse :

1. conserver les 20 états musculaires et de fatigue de Ding;
2. utiliser la mécanique réduite avec seulement l'angle du pédalier
   $\theta$ et sa vitesse $\omega$, sans imposer $\omega$ constante;
3. construire le NLP en SX;
4. utiliser un OCP d'un cycle et transférer le primal par shift cyclique,
   projection sur les bornes et projection mécanique;
5. compiler une seule fois les fonctions du NLP, puis rendre paramétriques
   l'état initial, la cible angulaire absolue et les bornes mobiles;
6. employer IPOPT/MUMPS comme chemin robuste certifié, et MadNLP/MUMPS comme
   candidat plus rapide lorsque sa politique de reprise au RHO courant est
   activée et validée;
7. inclure la relation de force passive et raffiner l'intégration du calcium;
8. auditer chaque RHO indépendamment du statut retourné par le solveur.

Cette réponse contient une réserve essentielle : les meilleurs temps 100 RHO
publiés utilisent encore la collocation Radau de degré 3. Cette transcription
est une bonne baseline de performance, mais elle n'est plus la cible
scientifique, car elle sous-estime le calcium périodique testé d'environ
`6.39 %`. Les résultats physiologiques définitifs doivent être recertifiés
avec une intégration du calcium convergée.

## 1. Principe directeur : corriger plutôt que reproduire

Le code historique reste utile comme point de comparaison logiciel. Il ne
constitue pas une vérité physique ni numérique. Dès qu'une approximation ou
une erreur est identifiée, le benchmark doit comparer les nouvelles méthodes
sur le **problème corrigé**, et non chercher à retrouver l'ancienne valeur de
coût.

Une différence avec l'ancienne référence est acceptable, et même attendue, si
elle provient d'une amélioration vérifiée de l'un des éléments suivants :

- équations musculaires;
- force passive;
- résolution temporelle du calcium;
- fermeture du contact mécanique;
- application des bornes aux points internes;
- définition absolue de l'angle terminal;
- audit de faisabilité physique.

L'ancienne solution ne sert alors qu'à localiser l'effet de la correction.
Elle ne doit jamais être utilisée comme oracle d'acceptation.

### 1.1 Trois niveaux de statut

| Statut | Signification | Usage permis |
|---|---|---|
| Historique | Résultat reproductible sur une ancienne transcription | Comprendre les développements et mesurer un gain logiciel |
| Certifié numérique | Chaîne RHO faisable avec audits indépendants | Comparer robustesse et temps sur cette transcription précise |
| Certifié scientifique | Modèle corrigé et étude de convergence temporelle réussie | Interpréter fatigue, coût et patrons de stimulation |

Le statut « solveur convergé » n'est pas un quatrième niveau. Sans audit, il
ne suffit pas à valider une fenêtre, encore moins une chaîne d'endurance.

## 2. Problème recommandé

### 2.1 États musculaires

Pour chacun des quatre muscles, conserver exactement les cinq états de Ding :

```math
x_m=
\begin{bmatrix}
C_{N,m} & F_m & A_m & \tau_{1,m} & K_{M,m}
\end{bmatrix}^{\mathsf T}.
```

Le modèle musculaire contient donc 20 états. La réduction recommandée ne
supprime aucun état de calcium, de force ou de fatigue.

### 2.2 Force passive

La relation de force utilisée dans le NLP doit inclure explicitement le terme
passif prévu par le modèle :

```math
\dot F_m =
\left[
A_m^{\mathrm{eff}}(PW_m)
\frac{C_{N,m}}{K_{M,m}+C_{N,m}}
-
\frac{F_m}
{\tau_{1,m}+\tau_2\frac{C_{N,m}}{K_{M,m}+C_{N,m}}}
\right]
\left(f_{\ell,m}f_{v,m}+f_{\mathrm{passif},m}\right).
```

Un ancien chemin de mise à jour du modèle perdait l'activation de cette
relation. Ce comportement n'est plus une référence à reproduire. Toute
campagne où le terme passif est absent doit être étiquetée comme ablation et
ne peut pas être comparée directement à la méthode recommandée.

La règle de développement est simple : créer, copier, mettre à jour ou
réduire un modèle ne doit jamais modifier silencieusement l'activation de la
force passive. Un test doit vérifier cette invariance sur les chemins full et
reduced.

### 2.3 Calcium : séparer résolution des contrôles et intégration des états

Le calcium est raide par rapport au pas associé aux 30 stimulations :

```math
\tau_c=0.011\ \mathrm{s},
\qquad
\Delta t=\frac{1}{30}\ \mathrm{s}
\approx 3.03\,\tau_c.
```

Dans le cas périodique isolé déjà testé :

| Transcription | Calcium périodique |
|---|---:|
| Solution analytique | `0.162982158353` |
| ACADOS IRK, 4 étages et 5 sous-pas | `0.162982158637` |
| Collocation Radau degré 3 | `0.152573519058` |
| Collocation Radau degré 4 | `0.163718500354` |
| Collocation Radau degré 5 | `0.162953961548` |
| Collocation Radau degré 6 | `0.162982834340` |
| ACADOS ERK testé | `0.232903256` |

Radau degré 3 reproduit exactement sa propre transcription, mais cette
transcription sous-estime ici la valeur analytique de `6.39 %`. ERK est encore
moins fidèle. Retrouver l'un de ces deux résultats ne constitue donc pas une
validation du calcium.

Les erreurs relatives des degrés 4, 5 et 6 sont respectivement `+0.4518 %`,
`-0.01730 %` et `+0.000415 %`. Radau 4 est donc un témoin coût-précision utile,
mais il ne satisfait pas le seuil scientifique de `0.1 %`. Radau 5 est le
premier degré testé qui le satisfait; Radau 6 sert de témoin de raffinement.

Le premier gate couplé IPOPT/reduced sur un RHO a convergé physiquement pour
les trois degrés. Radau 4, 5 et 6 ont demandé respectivement `10.50 s`,
`18.05 s` et `32.78 s` sur le même Mac non compilé à un thread. Entre Radau 5
et 6, l'écart valait `0.0316 %` sur la fatigue exécutée et `0.1022 %` sur son
AUC. Le gate Linux apparié sur cinq RHO confirme que cet écart ne disparaît
pas : selon le solveur, il atteint `0.343–0.398 %` sur la fatigue et
`0.580–0.645 %` sur l'AUC. Radau 5 reste donc un candidat, pas une méthode
certifiée.

La cible recommandée est de conserver 30 décisions de PW par cycle tout en
raffinant l'intégration des états entre deux décisions. Deux voies sont
pertinentes :

- collocation Radau d'ordre supérieur, actuellement degré 5 pour les audits
  IPOPT/MadNLP;
- sous-pas IRK internes, sans ajouter artificiellement de variables de
  contrôle.

Le choix final doit venir d'une étude de convergence. Les seuils proposés pour
la prochaine certification sont :

- erreur relative du calcium périodique isolé inférieure à `0.1 %`;
- variation du coût de fatigue et de son AUC inférieure à `0.1 %` lors du
  raffinement suivant;
- absence de nouvelle violation des bornes entre les nœuds.

Ces seuils sont des critères de recette proposés. Ils doivent apparaître dans
les artefacts et ne doivent pas être remplacés par un simple accord avec
Radau degré 3.

### 2.4 Mécanique réduite sans cadence imposée

La formulation rapide optimise seulement :

```math
x_{\mathrm{mec}}^{\mathrm{red}}
=
\begin{bmatrix}
\theta & \omega
\end{bmatrix}^{\mathsf T},
\qquad
q=\Phi(\theta),
\qquad
\dot q=T(\theta)\omega.
```

La dynamique projetée est :

```math
T^{\mathsf T}M(\Phi)T\,\dot\omega
=
T^{\mathsf T}
\left(
\tau_{\mathrm{muscle}}
+\tau_{\mathrm{ext}}
-h(\Phi,T\omega)
-M(\Phi)\dot T\,\omega
\right).
```

Cette réduction conserve donc les variations de cadence. Elle ne remplace pas
la dynamique par $\dot q$ constant. Sur 100 RHO corrigés, IPOPT full et
reduced diffèrent d'environ `0.1 %` en fatigue, tandis que la médiane chaude
est accélérée d'environ `4.5x`. Le noyau mécanique isolé est environ `40x`
plus rapide.

La formulation full reste obligatoire comme contrôle scientifique périodique,
mais elle n'est pas le choix de production tant que la réduction passe les
comparaisons appariées de coût, AUC, capacité finale et patrons de stimulation.

### 2.5 Conditions terminales et absence de drift

L'angle terminal doit être défini par rapport à une cible absolue :

```math
\theta_N=\theta_{\mathrm{origine}}-2\pi k,
```

où $k$ est l'indice absolu du cycle. Il ne doit pas être reconstruit à partir
de la solution terminale du cycle précédent. Cette définition empêche
l'accumulation d'un drift pourtant admissible fenêtre par fenêtre.

La cadence et les 20 états de Ding sont transférés d'une fenêtre à l'autre.
Les positions et vitesses mécaniques full doivent être reprojetées sur la
variété de contact. Les bornes de cadence doivent être vérifiées aux points
internes de la transcription, pas uniquement aux nœuds de tir.

## 3. Pile d'accélération recommandée

Les leviers sont à appliquer dans cet ordre, car les premiers réduisent aussi
le risque numérique des suivants.

### 3.1 Réduire la mécanique

Passer de 6 à 2 états mécaniques diminue la taille du KKT tout en conservant
les 20 états musculaires. C'est le gain structurel le mieux validé : environ
`4.5x` sur la médiane chaude IPOPT au palier 100 RHO.

### 3.2 Utiliser SX

Sur ce problème, SX réduit de `57.5 %` à `60.5 %` la médiane chaude par
rapport à MX, à objectifs identiques à environ `5e-11` près. Le temps de
construction plus long est payé une fois et ne doit pas être inclus dans le
temps chaud du RHO.

Tous les solveurs de la campagne active doivent donc utiliser la même
représentation SX. MX reste réservé aux expériences qui l'exigent réellement,
par exemple certains horizons complets exploratoires.

### 3.3 Compiler une fois et paramétrer ce qui change

La bibliothèque compilée doit être réutilisée pendant toute la chaîne RHO.
Les données suivantes sont des paramètres runtime, pas des raisons de
reconstruire le graphe :

- état initial musculaire et mécanique;
- angle terminal absolu;
- bornes mobiles;
- cibles de régularisation;
- paramètres de la fenêtre courante.

Les artefacts doivent prouver qu'un seul hash de bibliothèque est utilisé et
que les vecteurs de bornes changent réellement entre les RHO.

### 3.4 Warm-start commun

Le transfert recommandé est volontairement simple :

1. décaler cycliquement le primal convergé;
2. recaler la phase sur l'angle absolu du nouveau cycle;
3. borner les PW dans $[pd0,600\,\mu\mathrm{s}]$;
4. projeter les états mécaniques sur les bornes et, en full, sur le contact;
5. conserver la continuité des 20 états Ding;
6. recalculer les défauts avant l'appel solveur.

Le rollout IRK concurrent a ajouté environ `0.245 s` par transfert sans
prolonger le préfixe ACADOS. Il reste un diagnostic opt-in, pas le warm-start
par défaut. Une solution non convergée ne doit jamais alimenter le RHO suivant.
Le retry doit repartir du dernier checkpoint certifié et résoudre à nouveau
le **même** RHO.

### 3.5 Choisir le solveur selon le niveau de garantie

| Besoin | Solveur recommandé | État actuel |
|---|---|---|
| Chaîne robuste de 100 RHO | IPOPT/MUMPS reduced, SX, compilation persistante | `100/100`, environ `1.0 s` chaud sur Radau 3 |
| Collocation du calcium raffinée | MadNLP/MUMPS reduced, Radau 5 | `5/5`, environ `1.99 s` chaud; endurance à recertifier |
| Contrôle indépendant de l'optimum reduced | FATROP/collocation reduced | `100/100`, mais plus lent qu'IPOPT reduced |
| Cible sous-seconde | ACADOS IRK | Environ `0.4–0.6 s`, mais préfixe physique limité à 13 RHO |

MUMPS est le backend portable retenu pour IPOPT et MadNLP. PARDISO n'a pas
apporté de gain à MadNLP et reste archivé. FATROP full reste bloqué par
l'identification de structure des gaps. Alpaqa n'est pas fonctionnel sur cette
formulation.

Le meilleur résultat chaud brut n'est pas nécessairement la meilleure
méthode. ACADOS est actuellement le plus rapide par fenêtre convergée, mais
IPOPT reduced est le choix robuste pour une endurance complète. MadNLP devient
le meilleur candidat pour la cible scientifique raffinée, car Radau 5 y a
convergé là où IPOPT a stagné sur un RHO pourtant primalement faisable.

## 4. Protocole de certification

### 4.1 Gates séquentiels

Toute modification du modèle, du maillage ou du solveur doit passer les
paliers `5`, `30`, puis `100` RHO. Le palier suivant n'est lancé que si le
préfixe physique strict couvre entièrement le précédent.

Les cas full et reduced d'un même solveur s'exécutent sur la même machine,
successivement, pour limiter la variabilité matérielle. Les familles de
solveurs peuvent s'exécuter sur des machines CI distinctes en parallèle.

### 4.2 Audit indépendant par RHO

Une fenêtre est acceptée seulement si tous les contrôles suivants passent :

- statut natif de convergence;
- résidu primal et dynamique inférieur au seuil déclaré;
- états et contrôles dans leurs bornes;
- contact mécanique en position et vitesse;
- angle terminal absolu;
- continuité de $\omega$ et des 20 états Ding;
- bornes vérifiées aux stages internes ou par réintégration dense;
- solution finie et absence de valeur obsolète provenant d'un RHO précédent.

L'audit utilise les unités physiques. Les résidus scalés restent utiles au
diagnostic du solveur, mais ne remplacent pas la faisabilité physique.

### 4.3 Comparaison scientifique

Comparer au minimum :

- durée de construction, compilation et préparation initiale;
- durée solveur et murale de chaque RHO;
- médiane, P90 et maximum des temps chauds;
- coût total et fatigue exécutée;
- AUC de fatigue pour chacun des quatre muscles;
- capacité finale $A/A_{\mathrm{scale}}$ par muscle;
- patrons de PW aux cycles 10, 30 et 100;
- variations de PW entre deux cycles;
- préfixe physique strict et premier RHO en échec.

Les patrons de PW peuvent basculer entre plusieurs ensembles actifs presque
équivalents. Une différence ponctuelle de PW n'invalide donc pas seule une
solution; elle doit être interprétée avec le couple, la phase, le coût et les
états de fatigue.

## 5. Ce qui ne doit plus servir de cible

- une exécution sans force passive alors que le modèle la prévoit;
- Radau degré 3 considéré comme vérité pour le calcium;
- ERK avec le maillage actuel;
- des bornes vérifiées uniquement aux nœuds de tir;
- un angle terminal relatif au cycle précédent;
- une fenêtre postérieure à un RHO échoué présentée comme endurance valide;
- un accord avec un ancien coût utilisé comme seul critère de validation;
- une accélération obtenue en changeant simultanément modèle, transcription et
  solveur sans ablations appariées.

Ces configurations peuvent rester dans l'historique ou dans une campagne
d'ablation. Elles doivent être étiquetées explicitement et ne doivent pas
porter le nom `reference` dans les nouveaux artefacts. Les nouveaux noms
devraient décrire la méthode, par exemple `legacy-radau3`,
`scientific-radau5` ou `irk-refined`.

Les profils CLI `scientific-radau4`, `scientific-radau5` et
`scientific-radau6` sont des contrats verrouillés : `periodic_node`, couple
constant, SX, collocation Radau au degré annoncé et contraintes initiales
actives. Une surcharge contradictoire est refusée. Seul Radau 5 porte le statut
`candidate`; Radau 4 et 6 sont des diagnostics de raffinement. Le nom du profil
ne remplace pas la certification de la fatigue, de l'AUC et des bornes internes.

Le premier gate Linux 5 RHO du
[run 30748390517](https://github.com/mickaelbegon/cocofest/actions/runs/30748390517)
n'a **pas** certifié Radau 5. MadNLP/MUMPS converge sur `5/5`, mais l'écart
Radau 5--6 atteint `0.3977 %` sur la fatigue exécutée et `0.6452 %` sur l'AUC,
au-dessus du seuil provisoire de `0.1 %`. IPOPT/Radau 5 s'arrête au préfixe
strict `1/5` : le RHO 2 est primalement faisable, mais atteint 2 000 itérations.

Le second gate du
[run 30750686602](https://github.com/mickaelbegon/cocofest/actions/runs/30750686602)
désactive le transfert des duals et raffine d'abord la transcription cible.
IPOPT/Radau 5 passe alors de `1/5` à `5/5`; le RHO 2 tombe de 2 000 à 140
itérations. Le changement corrige donc un problème de warm-start, mais ne
change pas la conclusion scientifique :

| Solveur, reduced | R4--R5 fatigue | R4--R5 AUC | R5--R6 fatigue | R5--R6 AUC |
|---|---:|---:|---:|---:|
| IPOPT/MUMPS | `0.0804 %` | `0.00374 %` | `0.3431 %` | `0.5797 %` |
| MadNLP/MUMPS | `0.0954 %` | `0.01744 %` | `0.3977 %` | `0.6452 %` |

Les PW R5 et R6 diffèrent surtout au premier cycle : l'écart RMS sur les 120
PW vaut environ `11.1–11.4 µs`, avec un maximum de `105–106 µs` porté par le
Biceps. Il tombe ensuite autour de `1.2–3.0 µs` RMS, mais la fatigue accumule
la différence de recrutement. La répétition du même patron avec IPOPT et
MadNLP indique un changement de branche optimale ou une sensibilité de la
transcription couplée, et non un artefact propre à un solveur.

Le contrôle full/reduced Radau 5 affine le diagnostic. MadNLP obtient des
fatigues pratiquement identiques (`0.00194 %` d'écart), tandis qu'IPOPT trouve
en full une branche plus basse de `0.400 %` que son reduced. Les deux solveurs
s'accordent pourtant en reduced à `0.0205 %`. La réduction mécanique n'est
donc pas mise en défaut par le résultat MadNLP; il faut d'abord transférer et
réintégrer les mêmes PW entre full, reduced, R5 et R6 avant d'attribuer les
écarts au modèle.

Le statut rouge du run vient uniquement d'une erreur du contrôle CI : le gate
exigeait à tort une bibliothèque C pour le cas scientifique full, alors que
cette ablation chronomètre volontairement les évaluateurs interprétés. Les
résultats numériques et les artefacts sont complets. Le palier 30 reste bloqué
jusqu'à séparation de l'erreur de transcription et du changement de bassin
optimal.

Le prochain gate produit cette séparation directement. Chaque préfixe Radau
scientifique est exporté sans perte dans `validated-rho-trajectory.npz`, puis
ses PW sont rejouées avec une intégration continue DOP853 commune
(`rtol=1e-11`, `atol=1e-13`). Le rollout ne se recale pas sur les états de
collocation aux nœuds. Il rapporte le coût de fatigue, l'AUC, les quatre
muscles et l'écart aux états transcrits. Si les PW R5 et R6 restent différentes
mais donnent le même classement sous DOP853, l'écart vient principalement du
bassin de recrutement; si le classement change fortement, la transcription
reste le facteur dominant.

La comparaison DOP853 qui décide R5/R6 doit être lue d'abord en mécanique
reduced, où la géométrie du pédalier est intégrée dans les coordonnées. En
full, le rollout non projeté est un audit sévère du drift entre nœuds; ses
métriques de fatigue ne sont comparables à reduced que si l'écart mécanique
et la contrainte d'axe restent dans leurs tolérances.

## 6. Prochaine campagne recommandée

La prochaine comparaison utile ne consiste pas à accélérer davantage la
transcription historique. Elle doit d'abord fixer la nouvelle cible
scientifique commune :

1. force passive active et testée après toute copie ou mise à jour du modèle;
2. mécanique reduced validée contre full sur les mêmes PW et états Ding;
3. 30 décisions de PW, mais calcium intégré avec Radau 5 ou IRK sous-pas;
4. étude courte Radau 3/4/5/6, transfert croisé des solutions, puis
   réintégration dense commune;
5. IPOPT/MUMPS et MadNLP/MUMPS comparés sur cette même transcription;
6. ACADOS comparé seulement après alignement des contraintes internes;
7. paliers 5, 30 et 100, puis prolongation vers 300 et 1000 RHO pour atteindre
   un échec réellement attribuable à la fatigue.

La question de performance devient alors : « quelle méthode résout le plus
vite le problème corrigé avec le même niveau d'erreur? », et non « quelle
méthode reproduit le mieux l'ancienne référence? ».

## 7. Reproductibilité

Le workflow de benchmark est
[`cycling_solver_benchmark_linux.yml`](../../.github/workflows/cycling_solver_benchmark_linux.yml).
Il doit enregistrer dans chaque artefact les SHA complets de Cocofest,
Bioptim et ACADOS, les options du solveur, la transcription du calcium et
l'état de la force passive.

Les commandes détaillées, les versions historiques, les tableaux complets et
les liens vers les campagnes CI sont conservés dans
l'[historique des développements](development_history.md). Les décisions
actives doivent être mises à jour ici seulement après une campagne appariée et
certifiée.
