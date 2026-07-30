# Benchmark des solveurs du RHO de pédalage FES

Ce document est la source de référence pour le benchmark de fatigue du
pédalage FES. Il regroupe le problème scientifique, les formulations
mathématiques, la discrétisation, les stratégies de warm-start, les critères
de validation, l’organisation GitHub Actions et l’interprétation des résultats.

La campagne courante est volontairement **SX-only** et compare IPOPT/MUMPS,
MadNLP/MUMPS et ACADOS, en mécanique full et reduced. PARDISO est archivé :
il n’a apporté aucun gain à MadNLP. Fatrop est également sorti de l’endurance
courante, car sa formulation full échoue encore en SX lors de l’identification
de la structure des gaps. Alpaqa reste documenté comme intégration non
fonctionnelle.

Les résultats historiques d’endurance proviennent principalement du
[run Linux 30487321536](https://github.com/mickaelbegon/cocofest/actions/runs/30487321536).
La preuve comparative SX/MX vient du
[run Linux 30475768127](https://github.com/mickaelbegon/cocofest/actions/runs/30475768127).
Ils servent de référence avant la nouvelle campagne séquentielle 5, 30 puis
100 RHO.

## Résumé opérationnel

- L’objectif est uniquement de minimiser la fatigue musculaire.
- Le problème courant utilise un couple externe nul, `0 N.m`.
- Un OCP contient un cycle de pédalage et 30 stimulations par muscle.
- Un benchmark de 100 RHO résout donc 100 OCP successifs, et non un OCP de
  100 cycles.
- Les quatre muscles fournissent 20 états de Ding; la mécanique complète porte
  le total à 26 états et la mécanique réduite à 22.
- La référence NLP utilise une collocation Radau de degré 3.
- Tous les OCP et warm-starts IPOPT de la campagne active utilisent SX. Le
  runner refuse explicitement une demande MX.
- MadNLP utilise exclusivement MUMPS, transmis à libMad sous le nom typé
  exact `MumpsSolver`. La CI échoue si libMad signale une option inconnue.
- PARDISO, Fatrop et RK4 ne font pas partie de la campagne active. Leurs
  anciens résultats sont conservés comme diagnostics historiques.
- MadNLP reste interprété : la compilation C est refusée par l’interface
  Bioptim/CasADi épinglée et n’est pas comptée comme une variante valide.
- La convergence du solveur ne suffit pas : chaque RHO est soumis à un audit
  indépendant de faisabilité physique avec un seuil de `1e-5`.
- La mécanique réduite est plus rapide, mais le grand écart de fatigue observé
  n’est pas encore une réduction physiologique démontrée. Les problèmes full
  et reduced ne sont actuellement pas mécaniquement équivalents.

## 0. Versions Bioptim et reproductibilité

Le benchmark GitHub Actions n’utilise pas le paquet Conda
`bioptim==3.4.0` mentionné dans les instructions d’installation générales de
Cocofest. Il clone directement le fork
[`mickaelbegon/BiorbdOptim`](https://github.com/mickaelbegon/BiorbdOptim)
et effectue chaque checkout par SHA complet. Le SHA est la référence
reproductible; les noms de branche servent uniquement à documenter la
provenance humaine.

### 0.1 Campagne d’endurance principale

| Composant | Version Bioptim réellement utilisée |
|---|---|
| Construction et certification des seeds | `3523f1745e315f07761159d7e06bd2d876026704` |
| IPOPT full/reduced | `3523f1745e315f07761159d7e06bd2d876026704` |
| MadNLP/MUMPS full/reduced | `3523f1745e315f07761159d7e06bd2d876026704` |
| ACADOS full/reduced et variantes | `3523f1745e315f07761159d7e06bd2d876026704` |

Ce commit date du 27 juillet 2026 et porte le correctif
`Fix Fatrop physical bound auditing`. Le workflow lui associe le libellé de
provenance `codex/fatrop-cocofest-benchmark`, mais ne fait jamais un checkout
flottant de cette branche.

### 0.2 Fatrop historique et patchs Bioptim

L’ancienne campagne Fatrop transformait ce checkout, dans cet ordre :

1. [`bioptim-fatrop-c-compile-plugin-case.patch`](../../.github/patches/bioptim-fatrop-c-compile-plugin-case.patch),
   qui corrige la casse du nom du plugin lors du chargement des évaluateurs C;
2. [`bioptim-fatrop-scaled-gaps.patch`](../../.github/patches/bioptim-fatrop-scaled-gaps.patch),
   qui normalise les gaps, les helpers de collocation et les transitions de
   phase pour préserver le bloc identité exigé par Fatrop.

Le second patch est un port minimal du commit Bioptim
`70c384517af48502e5e1bda6c48beb4c515cb8a1`
(`fix: support scaled constraints with FATROP`, 28 juillet 2026). La branche
`codex/fatrop-scaling-audit` complète n’est pas utilisée, car elle ne contient
pas les correctifs plus récents présents dans le SHA de production.

La version effective des résultats Fatrop historiques doit donc être
identifiée par le triplet :

```text
Bioptim 3523f1745e315f07761159d7e06bd2d876026704
+ bioptim-fatrop-c-compile-plugin-case.patch
+ bioptim-fatrop-scaled-gaps.patch
```

Fatrop n’est plus installé ni exécuté par la campagne SX-only. Le reduced SX
a résolu le smoke d’un RHO, mais le full SX s’arrête avant le premier RHO avec
une incohérence de structure de \(A\). Le réintroduire exigerait de corriger
cette structure dans Bioptim puis de valider full et reduced en SX; revenir à
MX contredirait le protocole numérique courant.

### 0.3 Screens historiques séparés

Le mode `cycles=screen` n’appartient pas à la campagne d’endurance et conserve
deux intégrations historiques :

| Screen | Commit Bioptim | Provenance |
|---|---|---|
| MadNLP/MUMPS, mode `cycles=screen` | `346eb1d445e6ba67010b96c6f16ba830185119e7` | `codex/madnlp-integration-master` |
| Alpaqa/PANOC | `d84e7e43534360fc048e0be26a3bd69a2abc2d77` | `codex/alpaqa-integration` |

Ces résultats ne doivent pas être mélangés avec ceux de la campagne principale
sans mentionner ce changement de Bioptim.

Le workflow source de vérité est
[`cycling_solver_benchmark_linux.yml`](../../.github/workflows/cycling_solver_benchmark_linux.yml).
Chaque résultat exporte également `BIOPTIM_BENCHMARK_COMMIT` et
`COCOFEST_BENCHMARK_COMMIT`, afin qu’un artefact reste attribuable même après
une modification ultérieure du README.

## 1. Problème de commande optimale

### 1.1 États musculaires

Pour chaque muscle \(m\), le modèle de Ding avec fatigue utilise cinq états :

\[
x_m =
\begin{bmatrix}
C_{N,m} & F_m & A_m & \tau_{1,m} & K_{\mathrm M,m}
\end{bmatrix}^{\mathsf T}.
\]

Ils représentent respectivement le complexe calcium-troponine, la force, la
capacité de production de force et deux paramètres dynamiques qui évoluent avec
la fatigue. Avec quatre muscles,

\[
n_{\mathrm{Ding}} = 4 \times 5 = 20.
\]

La dynamique de force peut être écrite sous la forme :

\[
\dot F_m =
\left[
A_m^\mathrm{eff}(PW_m)
\frac{C_{N,m}}{K_{\mathrm M,m}+C_{N,m}}
-
\frac{F_m}
{\tau_{1,m}+\tau_2\frac{C_{N,m}}{K_{\mathrm M,m}+C_{N,m}}}
\right]
\left(f_{\ell,m} f_{v,m}+f_{\mathrm{passif},m}\right),
\]

où l’effet de la largeur d’impulsion est

\[
A_m^\mathrm{eff}(PW_m)
=
A_m\left[
1-\exp\left(-\frac{PW_m-pd0_m}{pdt_m}\right)
\right].
\]

Les trois états lents de fatigue suivent :

\[
\dot A_m
=
-\frac{A_m-A_{\mathrm{scale},m}}{\tau_{\mathrm{fat},m}}
+\alpha_{A,m}F_m,
\]

\[
\dot \tau_{1,m}
=
-\frac{\tau_{1,m}-\tau_{1,\mathrm{rest},m}}{\tau_{\mathrm{fat},m}}
+\alpha_{\tau_1,m}F_m,
\]

\[
\dot K_{\mathrm M,m}
=
-\frac{K_{\mathrm M,m}-K_{\mathrm M,\mathrm{rest},m}}{\tau_{\mathrm{fat},m}}
+\alpha_{K_m,m}F_m.
\]

Dans les paramètres courants, \(\alpha_A<0\), tandis que
\(\alpha_{\tau_1}>0\) et \(\alpha_{K_m}>0\).
La force élevée et répétée fait diminuer la capacité normalisée
\(A_m/A_{\mathrm{scale},m}\).

### 1.2 Largeurs d’impulsion

Chaque muscle possède 30 commandes de largeur d’impulsion par cycle :

\[
u_k =
\begin{bmatrix}
PW_{1,k} & PW_{2,k} & PW_{3,k} & PW_{4,k}
\end{bmatrix}^{\mathsf T},
\qquad k=0,\ldots,29.
\]

Les bornes physiques sont

\[
pd0_m \le PW_{m,k} \le 600\ \mu\mathrm{s}.
\]

Pour les paramètres Ding courants,

\[
pd0 \simeq 131.405\ \mu\mathrm{s}.
\]

`pd0` est le vrai zéro de recrutement du modèle. Une commande inactive doit
être fixée à `pd0`, et non à une largeur d’impulsion numérique nulle. Les seeds
chargés sont validés et tronqués explicitement dans cet intervalle; chaque
correction produit un warning.

### 1.3 Objectif de fatigue

L’objectif continu quadratique est

\[
J =
10\,000
\int_0^T
\sum_{m=1}^{4}
\left(
1-\frac{A_m(t)}{A_{\mathrm{scale},m}}
\right)^2
\,dt,
\]

où le poids de fatigue actif vaut \(10\,000\). Les autres composantes de coût
sont désactivées dans le benchmark :

\[
w_{\mathrm{force}} =
w_{\mathrm{contrôle}} =
w_{\mathrm{angle\ terminal}} =
w_{\dot q} = 0.
\]

Le solveur minimise donc la fatigue, pas la force, la charge électrique ou la
régularité des stimulations. Cette absence de régularisation autorise plusieurs
patrons localement optimaux et peut déplacer le partage Biceps/Triceps sans
modifier fortement l’objectif.

Deux métriques doivent être distinguées :

1. `executed_fatigue_objective`, qui réévalue le coût quadratique sur les
   cycles réellement exécutés;
2. `fatigue_auc_cycles`, définie par

   \[
   \mathrm{AUC}_{\mathrm{fatigue}}
   =
   \int
   \sum_m
   \left(1-\frac{A_m}{A_{\mathrm{scale},m}}\right)
   d(\mathrm{cycle}).
   \]

Le coût quadratique amplifie les muscles fortement fatigués. Un rapport de
coût de 6 ou 7 ne signifie donc pas nécessairement une AUC 6 ou 7 fois plus
grande.

## 2. Receding horizon

Le terme RHO désigne ici une résolution de l’OCP dans la séquence à horizon
glissant. Pour un cycle par OCP, la résolution \(r\) optimise

\[
\mathcal P_r:
\quad
\min_{z_r} f(z_r;p_r)
\]

sous

\[
g(z_r;p_r)=0,
\qquad
\underline g_r \le h(z_r;p_r)\le\overline g_r,
\qquad
\underline z_r\le z_r\le\overline z_r.
\]

Le vecteur \(p_r\) regroupe les informations qui changent sans modifier le
graphe symbolique :

- l’état exécuté du cycle précédent;
- la cible angulaire absolue;
- les bornes de continuité et les bornes terminales;
- les données nécessaires au décalage des stimulations;
- les états de fatigue accumulés.

La cible terminale est absolue :

\[
\theta_{\mathrm{cible}}(r)
=
\theta_0+r\Delta\theta,
\qquad
\Delta\theta=-2\pi,
\]

avec

\[
\left|\theta(T)-\theta_{\mathrm{cible}}(r)\right|
\le 0.002\ \mathrm{rad}.
\]

Cette définition empêche un drift de même signe d’un cycle au suivant. Une
cible définie à partir du terminal précédent aurait autorisé une accumulation
de petites erreurs, même si chaque RHO respectait localement sa tolérance.

Le warm-start primal s’écrit schématiquement :

\[
z_{r+1}^{(0)}
=
\Pi_{\mathcal B_{r+1}}
\left(
\mathcal S z_r^\star
\right),
\]

où \(\mathcal S\) décale la trajectoire d’un cycle et
\(\Pi_{\mathcal B_{r+1}}\) la projette dans les nouvelles bornes. Les états de
fatigue sont continus; ils ne sont pas remis à leur valeur reposée.

Le benchmark autorise deux échecs consécutifs afin de distinguer un échec
isolé d’une perte persistante de robustesse. Le `validated_cycles` utilisé
pour l’endurance reste néanmoins le préfixe strict avant le premier RHO
invalide.

## 3. Mécanique complète

La mécanique complète possède trois coordonnées généralisées :

\[
q\in\mathbb R^3,
\qquad
\dot q\in\mathbb R^3.
\]

Avec les 20 états musculaires :

\[
n_x^{\mathrm{full}}=20+3+3=26.
\]

Les équations contraintes sont :

\[
M(q)\ddot q+h(q,\dot q)
=
\tau_{\mathrm{muscles}}(q,\dot q,F)
+\tau_{\mathrm{ext}}
+J_c(q)^{\mathsf T}\lambda,
\]

\[
J_c(q)\ddot q+\dot J_c(q,\dot q)\dot q=0.
\]

La seconde équation impose une accélération de contact nulle. Elle n’impose
pas à elle seule :

\[
c(q)=0,
\qquad
J_c(q)\dot q=0.
\]

Ces deux conditions doivent être vraies au départ. Sinon, une erreur de
position ou de vitesse de contact peut être propagée par une dynamique
parfaitement faisable au niveau accélération.

Dans l’implémentation courante, les contraintes explicites de position et de
vitesse du centre du pédalier ne sont ajoutées au nœud initial que lorsque
`enforce_start_constraints=True`. Le profil du benchmark les désactive
actuellement. C’est une différence centrale avec la mécanique réduite.

## 4. Mécanique réduite

Les deux contraintes holonomes réduisent les trois coordonnées mécaniques à
un degré de liberté. La formulation réduite utilise l’angle physique non
enroulé du pédalier \(\theta\) et sa vitesse \(\omega\) :

\[
x_{\mathrm{mécanique}}^{\mathrm{reduced}}
=
\begin{bmatrix}\theta & \omega\end{bmatrix}^{\mathsf T},
\qquad
n_x^{\mathrm{reduced}}=20+2=22.
\]

Elle n’impose pas une vitesse constante :

\[
\dot\theta=\omega.
\]

### 4.1 Construction de la variété

Pour chaque angle \(\theta\), trois équations déterminent \(q(\theta)\) :

1. position horizontale du centre du pédalier;
2. position verticale du centre du pédalier;
3. orientation du vecteur centre-main correspondant à l’angle physique.

La solution périodique est représentée par

\[
q(\theta)
=
w\,s(\theta)+
\sum_{k=0}^{K}
\left(
a_k\cos(k\theta)+b_k\sin(k\theta)
\right),
\]

où \(w\,s(\theta)\) porte l’enroulement non périodique de la coordonnée du
pédalier. Les dérivées sont analytiques :

\[
\dot q = T(\theta)\omega,
\qquad
T(\theta)=\frac{dq}{d\theta},
\]

\[
\ddot q
=
T(\theta)\dot\omega
+\frac{d^2q}{d\theta^2}\omega^2.
\]

### 4.2 Projection dynamique

La projection tangentielle donne

\[
M_{\mathrm{eff}}(\theta)=T^{\mathsf T}M(q(\theta))T,
\]

et

\[
\dot\omega
=
\frac{
\sum_m r_m(\theta)F_m
+r_{\mathrm{ext}}(\theta)\tau_{\mathrm{ext}}
-g_{\mathrm{eff}}(\theta)
-c_{\mathrm{eff}}(\theta)\omega^2
}{
M_{\mathrm{eff}}(\theta)
}.
\]

Les coefficients périodiques suivants sont tabulés puis ajustés par séries de
Fourier :

- inertie effective;
- gravité projetée;
- terme quadratique en vitesse;
- efficacité mécanique de chaque muscle;
- efficacité du couple externe;
- longueurs musculaires normalisées;
- vitesses musculaires par unité de \(\omega\).

Les lois force-longueur, force-vitesse et passive originales sont ensuite
évaluées avec ces géométries. Le réseau musculaire ou le modèle Ding n’est pas
remplacé par un surrogate.

### 4.3 Validation de la réduction

Une validation directe sur 500 points admissibles a donné :

- résidu de contact maximal : environ `1.5e-9 m`;
- erreur médiane d’accélération : environ `6.1e-4 rad/s²`;
- erreur maximale d’accélération : environ `9.6e-3 rad/s²`;
- erreur relative P95 : environ `1.6e-5`;
- erreurs force-longueur, force-vitesse et passive : sous `6e-9`;
- accélération mécanique isolée : environ `40x` plus rapide.

L’approximation mécanique est donc trop précise pour expliquer à elle seule
un rapport de coût de fatigue proche de 7.

Commande de validation :

```bash
python examples/fes_multibody/cycling/validate_reduced_cycling_dynamics.py \
  --samples 181 \
  --kinematic-order 12 \
  --dynamics-order 12 \
  --validation-samples 500 \
  --external-crank-torque 0.0 \
  --casadi-profile \
  --casadi-repeats 1000 \
  --output-profile result/reduced-cycling.npz \
  --output-json result/reduced-cycling-validation.json
```

## 5. Pourquoi full et reduced ne sont pas encore équivalents

### 5.1 Erreur de contact du seed full

Dans le seed full commun du run `30487321536` :

- le résidu de position du centre du pédalier atteint `0.102 m`;
- son RMS atteint `0.072 m`;
- le résidu de vitesse planaire atteint `0.0376 m/s`;
- la projection sur la variété réduite corrige jusqu’à `0.575 rad`;
- le résidu de vitesse tangentielle atteint `3.23 rad/s`.

La réduction reconstruit toujours une posture sur \(c(q)=0\), tandis que la
formulation complète peut poursuivre la trajectoire hors variété. Les longueurs
musculaires, vitesses de contraction, bras de levier et forces nécessaires
sont alors différents.

Le benchmark impose maintenant \(c(q_0)=0\) et
\(J(q_0)\dot q_0=0\) au début de chaque RHO. Ces contraintes ne sont pas
dupliquées à tous les nœuds : elles seraient redondantes avec la dynamique
contrainte au niveau accélération et pourraient rendre le KKT déficient en
rang. En complément, chaque trajectoire full est projetée sur la variété
réduite. Le rapport contient désormais l’erreur maximale/RMS de configuration,
le résidu maximal/RMS de vitesse tangentielle et les traces physiques
\(\theta,\omega\). Une trajectoire qui dépasse `0.01 rad` ou `0.1 rad/s` est
marquée physiquement invalide, même si le solveur converge.

Le seed historique a aussi été construit à l’aide d’un modèle d’IK qui contient
des transformations fixes différentes de celles du modèle dynamique. Cette
différence géométrique est une autre source plausible de l’incompatibilité
initiale.

Un second défaut a été corrigé dans la construction du warm-start de
collocation. Radau degré 3 n’utilise pas quatre temps uniformes dans chaque
intervalle, mais

\[
\tau=[0,\ 0.1550510257,\ 0.6449489743,\ 1].
\]

L’IK et ses dérivées sont maintenant évalués sur ces temps physiques, y
compris les temps dupliqués entre le point Radau \(\tau=1\) et le nœud de tir
suivant. Le recentrage de \(\theta\) utilise également cette grille et applique
à \(\omega\) la dérivée temporelle de la correction, au lieu de modifier
l’angle seul.

### 5.2 Angle relatif contre angle physique

Dans la formulation complète, `q[2]` est une rotation articulaire relative.
Dans la formulation réduite, \(\theta\) est l’angle physique du vecteur
centre-main. Le long de la variété :

\[
\frac{dq_2}{d\theta}\in[0.591,\ 1.452].
\]

Borner `qdot[2]` et \(\omega\) avec le même intervalle numérique ne borne donc
pas la même cadence physique. De même, appliquer un couple constant sur
`q[2]` donne en coordonnée physique une efficacité modulée par
\(dq_2/d\theta\).

Ce point ne contribue pas par le couple externe dans le run courant, puisque
\(\tau_{\mathrm{ext}}=0\), mais il affecte les bornes de vitesse et les
diagnostics de phase.

### 5.3 Effet sur les stimulations et la fatigue

Au cycle 10, la solution full atteint près de `600 µs` pour Biceps et Triceps.
La solution reduced reste autour de `225 µs` et `218 µs`. Dans les seeds
communs, les maxima sont respectivement proches de `570/594 µs` contre
`217/208 µs`.

La non-linéarité

\[
1-\exp\left(-\frac{PW-pd0}{pdt}\right)
\]

rend les pics de PW beaucoup plus importants que ne le suggère une simple
comparaison des moyennes. La différence de recrutement se transforme
directement en force, puis en fatigue de \(A\), \(\tau_1\) et \(K_m\).

### 5.4 Expérience appariée requise

Avant d’interpréter le gain de fatigue comme physiologique, il faut :

1. construire le seed full par relèvement exact
   \(q=q(\theta)\), \(\dot q=T(\theta)\omega\);
2. imposer la position et la vitesse de contact au premier nœud de chaque RHO;
3. auditer les résidus de contact à tous les nœuds;
4. utiliser le même angle et la même vitesse physiques pour les bornes et la
   cible terminale;
5. rejouer les mêmes PW sur les deux mécaniques en intégrant les 20 états de
   Ding;
6. effectuer des initialisations croisées full vers reduced et reduced vers
   full.

Seul l’écart résiduel après ce test peut être attribué à la réduction
dynamique.

Les quatre premières briques sont maintenant engagées dans la CI :

- le seed reduced est résolu en premier;
- le seed full est initialisé avec le relèvement exact
  \(q(\theta),T(\theta)\omega\);
- les bornes mécaniques du seed full sont recadrées sans tronquer ce
  relèvement;
- les contraintes de contact initiales et les bornes de cadence physique sont
  actives pour les NLP;
- les phases, cadences et résidus de variété sont audités dans les coordonnées
  physiques.

Un smoke test local IPOPT/MUMPS à `0 N.m`, un RHO, a donné :

| Formulation | Coût de fatigue exécuté |
|---|---:|
| reduced | `3.599121` |
| full relevé et contraint | `3.604350` |

L’écart n’est plus que d’environ `0.15 %`, contre un facteur proche de 7 dans
le benchmark non apparié. Ce résultat local est très encourageant mais ne
remplace pas encore le test CI apparié de 30 puis 100 RHO.

Pour le full, la tolérance terminale portée par `q[2]` est réduite par
\(\min_\theta |dq_2/d\theta|\), afin qu’elle implique la tolérance physique
absolue demandée sur \(\theta\). Une contrainte terminale vectorielle
cross/dot plus directe a été prototypée, mais elle fait actuellement avorter
l’initialisation IPOPT dans la pile Bioptim/CasADi locale; elle reste donc
désactivée et la phase physique terminale demeure vérifiée a posteriori.

## 6. Discrétisation et taille du NLP

Le benchmark NLP utilise \(N=30\) intervalles et une collocation Radau de
degré \(d=3\). Pour un état de taille \(n_x\), le nombre approximatif de
variables d’état stockées est

\[
n_x\left[1+N(d+1)\right].
\]

En ajoutant \(4N=120\) commandes :

\[
n_z^{\mathrm{full}}
\approx 26(121)+120=3266,
\]

\[
n_z^{\mathrm{reduced}}
\approx 22(121)+120=2782.
\]

La réduction du nombre de variables est donc d’environ 15 %, insuffisante pour
expliquer seule le gain de temps proche de 2. Le principal gain vient de la
réduction de la complexité des dérivées mécaniques.

Pour chaque intervalle, les équations de collocation ont la forme

\[
X_{k,j}
=
X_k+h\sum_{r=1}^{d}a_{jr}f(X_{k,r},U_k),
\]

\[
X_{k+1}
=
X_k+h\sum_{r=1}^{d}b_r f(X_{k,r},U_k).
\]

Le classement temporel des variables place les blocs
\((X_k,U_k,X_{k,1},\ldots)\) de manière à préserver la structure par étage.
Il est utilisé pour IPOPT et MadNLP dans la campagne active. Les essais
Fatrop historiques utilisaient le même ordre. ACADOS conserve son organisation
native par étage.

## 7. Systèmes KKT et solveurs

Une itération de Newton ou de point intérieur conduit schématiquement au
système KKT :

\[
\begin{bmatrix}
H_L+D & J_g^{\mathsf T}\\
J_g & 0
\end{bmatrix}
\begin{bmatrix}
\Delta z\\
\Delta\lambda
\end{bmatrix}
=
-
\begin{bmatrix}
r_{\mathrm{dual}}\\
r_{\mathrm{primal}}
\end{bmatrix},
\]

où \(H_L\) est le Hessien du Lagrangien, \(J_g\) le Jacobien des contraintes
et \(D\) la contribution de barrière ou de régularisation.

### 7.1 IPOPT

IPOPT est la référence robuste. Il utilise une méthode primal-dual de point
intérieur, le Hessien exact et un solveur linéaire creux :

- MUMPS, disponible dans l’environnement standard;
- MA57, lorsque CoinHSL est disponible;
- les autres backends IPOPT ne font pas partie du benchmark courant.

Le warm-start de référence transfère le primal décalé et peut réutiliser les
multiplicateurs de bornes après contrôle de leur taille.

### 7.2 MadNLP

MadNLP suit également une méthode primal-dual de point intérieur avec Hessien
exact. Le backend retenu est MUMPS. L’API utilisateur garde le nom portable
`mumps`, mais libMad attend un nom de type Julia, sensible à la casse :
`MumpsSolver`. L’ancien passage de la chaîne `mumps` produisait :

```text
libMAD WARNING: option linear_solver is of unknown type mumps, ignoring
```

Le calcul continuait parce que MUMPS est aussi le défaut creux de MadNLP
0.9.2; le résultat était donc souvent correct, mais le choix demandé n’était
pas réellement certifié. L’adaptateur traduit maintenant
`mumps -> MumpsSolver`. Le smoke runtime vérifie cette valeur et chaque log
est rejeté si l’avertissement réapparaît.

PARDISO/MKL est laissé de côté. Sur le run `30511306081`, MUMPS était environ
30 % plus rapide en médiane full (`2.571 s` contre `3.668 s`) et 6 % plus
rapide en reduced (`1.155 s` contre `1.234 s`); les temps mur-à-mur étaient
également inférieurs (`350.0 s` contre `466.2 s`, puis `178.0 s` contre
`186.4 s`). Cette comparaison avait encore l’ancien avertissement côté
MUMPS, donc la campagne courante remesure explicitement `MumpsSolver`, sans
présenter PARDISO comme une variante active. Le commit libMad épinglé contient
toujours son code PARDISO, mais le workflow ne le sélectionne ni ne le
certifie. L’installation utilise le smoke `no_hsl_example`, qui exerce MUMPS
sans instancier PARDISO; `basic_problem` n’est volontairement plus lancé car
il testait aussi `PardisoMKLSolver`.

Le benchmark garde le transfert des multiplicateurs désactivé. Les blocs duaux
ne sont pas encore décalés structurellement avec le RHO; le primal décalé est
le warm-start fiable.

### 7.3 Fatrop — diagnostic historique, hors campagne SX-only

Fatrop exploite la structure d’OCP pour résoudre les systèmes linéaires par
une factorisation de type Riccati plutôt que comme une matrice KKT générique.
Cette exploitation requiert :

- un ordre temporel cohérent;
- des équations de transition identifiables;
- une structure de gap compatible avec la détection automatique.

Le scaling générique des états modifie le coefficient identité du prochain
état dans les gaps. Sans correction, Fatrop refuse la formulation full avec
le diagnostic « structure of A does not correspond » : le bloc associé à
\(x_{k+1}\) n’est plus l’identité attendue par le solveur structuré.

Fatrop relâche aussi les bornes de manière relative. Comme certaines capacités
de fatigue valent plusieurs milliers, un facteur relatif apparemment faible
peut produire un écart absolu supérieur au seuil physique. L’interface serre
les bornes transmises à Fatrop tout en conservant les bornes originales pour
l’audit indépendant.

Les anciens tests ont porté sur le commit Bioptim épinglé et le correctif
minimal de la branche `codex/fatrop-scaling-audit`. Chaque gap

\[
S z_{k+1}-\Phi(Sz_k,u_k)=0
\]

en

\[
z_{k+1}-S^{-1}\Phi(Sz_k,u_k)=0,
\]

de sorte que le Jacobien par rapport à \(z_{k+1}\) reste exactement
l’identité. La même transformation est appliquée aux helpers de collocation
et aux transitions de phase séquentielles. Le patch conserve simultanément
le resserrement physique des bornes présent dans notre commit Bioptim plus
récent; la branche complète n’est pas utilisée car elle est en retard de
28 commits sur cette base.

Les 7 tests Bioptim dédiés au patch passent localement, mais cela ne suffit
pas : Fatrop full échoue encore avec SX lors de la détection de structure,
alors que reduced SX a seulement passé un smoke 1/1. La campagne active
n’utilise donc pas Fatrop. Ses métriques MX antérieures restent utiles pour
localiser le coût des dérivées, mais ne sont pas mélangées aux résultats
SX-only.

### 7.4 ACADOS

ACADOS résout une suite de QP :

\[
\min_{\Delta z}
\frac12\Delta z^{\mathsf T}H_{\mathrm{GN}}\Delta z
+g^{\mathsf T}\Delta z
\]

sous la linéarisation des dynamiques et des contraintes. La référence utilise :

- SQP complet;
- Hessien de Gauss-Newton;
- intégration IRK;
- HPIPM;
- code généré une fois et réutilisé.

SQP-RTI n’effectue qu’une itération SQP par RHO. Il est potentiellement très
rapide, mais exige une trajectoire nominale déjà proche de la variété faisable.
Ce n’est pas encore le cas après plusieurs cycles.

La première stratégie testée est la voie native
`SQP_WITH_FEASIBLE_QP` :

1. direction `FEASIBILITY_QP` pour restaurer le primal;
2. bascule autorisée vers la direction nominale pour minimiser la fatigue.

Cette variante est ajoutée à l’écran ACADOS full/reduced, à côté du SQP
nominal, de `BYRD_OMOJOKUN`, de l’homotopie de bornes et de RTI. Elle réutilise
un seul OCP précompilé. Si elle ne restaure pas les résidus stricts, l’étape
suivante sera un second OCP précompilé avec dynamique dure et coût de proximité.
Les contrôles virtuels ne seront ajoutés qu’en dernier recours, avec un seuil
explicite sur leur norme avant acceptation.

La restauration locale actuelle `q/qdot + offsets PW` est explicitement
refusée avec la mécanique réduite `theta/omega`; elle n’est pas un OCP et ne
doit pas être interprétée comme une restauration reduced valide.

### 7.5 Alpaqa

Alpaqa combine une méthode de Lagrangien augmenté externe et PANOC à
l’intérieur. Cette méthode du premier ordre s’est montrée très sensible au
scaling et aux contraintes redondantes de collocation :

- aucun RHO validé dans les écrans achevés;
- limites de temps atteintes;
- infaisabilité encore au-dessus de `1e-5`;
- compilation C non supportée par le chemin CasADi actuel.

Alpaqa est donc exclu des benchmarks d’endurance. Le code de diagnostic reste
disponible pour reproduire les échecs.

## 8. SX, MX et compilation C

### 8.1 Graphes SX et MX

MX représente un graphe matriciel général. SX développe davantage les
opérations scalaires et peut produire un graphe plus grand à construire, mais
des évaluations plus simples à optimiser et compiler.

Sur le screen de 30 RHO du run `30475768127`, la médiane chaude est :

| Cas | MX médiane | SX médiane | Gain SX | Accélération |
|---|---:|---:|---:|---:|
| IPOPT full | 5.549 s | 2.328 s | 58.0 % | 2.38× |
| IPOPT reduced | 2.972 s | 1.263 s | 57.5 % | 2.35× |
| MadNLP/MUMPS full | 6.182 s | 2.584 s | 58.2 % | 2.39× |
| MadNLP/MUMPS reduced | 3.002 s | 1.187 s | 60.5 % | 2.53× |

Le gain est calculé par
\((T_\mathrm{MX}-T_\mathrm{SX})/T_\mathrm{MX}\), sur les RHO chauds 2 à 30.
Les quatre cas ont validé 30/30 RHO. Les statuts sont identiques et les écarts
absolus maximaux d’objectif par fenêtre valent respectivement
`8.6e-12`, `1.0e-11`, `2.46e-11` et `5.01e-11`. Les nombres d’itérations sont
identiques, sauf MadNLP reduced avec un écart maximal de trois itérations.
SX apporte donc ici un gain d’évaluation du même NLP, sans changement
scientifique mesurable de la solution.

Ce résultat justifie la règle de production : le profil IPOPT périodique, le
warm-up standard et les NLP IPOPT/MadNLP sont construits en SX. Le
raffinement IPOPT auxiliaire d’ACADOS reste SX lorsqu’il est demandé, mais la
CI ne le répète plus : le seed commun est déjà la solution certifiée d’un
IPOPT collocation SX, et deux runners ont été arrêtés après environ 150 s
pendant la construction redondante du raffinement full. Le script d’exécution
refuse `mx`, le JSON doit
contenir `use_sx=true`, et le rapport affiche le type de graphe. MX reste
uniquement une donnée historique de justification.

### 8.2 Évaluateurs C persistants

Les options sont :

```text
--ipopt-c-compile
```

Elles compilent les fonctions CasADi de coût, contraintes, gradient, Jacobien
et Hessien. Elles ne compilent pas le solveur non linéaire lui-même.

Le graphe reste constant durant les RHO. Les données mobiles sont fournies
numériquement :

\[
x_0,\quad l_x,\quad u_x,\quad l_g,\quad u_g.
\]

Un mode correctement persistant doit exporter :

```text
compiled_library_build_count == 1
compiled_library_reused == true
graph_rebuild_detected == false
runtime_bounds_changed == true
```

Le temps de compilation est compté dans le mur-à-mur, mais pas dans la médiane
chaude. Sur Apple Silicon, le coût initial peut dépasser le gain pour un petit
nombre de RHO. L’intérêt doit être évalué sur 100 RHO sous Linux.

Les campagnes principales full/reduced IPOPT utilisent la compilation
persistante quand `compile_nlp_evaluators=true`. Ce sont elles qui certifient
la compilation sur le problème d’endurance réellement mesuré. Pour chaque run
compilé multi-RHO, la CI exige :

```text
observed_solves == attempted_windows
runtime_bounds_changed == true
```

Cela vérifie que les bornes mobiles sont effectivement modifiées sans reconstruire
le graphe. Le tracker conserve également taille, `mtime` et SHA-256 de `nlp.c`;
la CI multi-RHO exige une seule version observée et sa réutilisation. Cette
preuve complète le contrôle d’identité du solveur CasADi sans inclure le coût
du hash à chaque RHO : le contenu n’est relu que si taille ou `mtime` changent.

L’ancienne ablation intégrée « interprété puis compilé sur cinq RHO » est
archivée. Elle reconstruisait deux OCP full supplémentaires après les cas full
et reduced, et le runner GitHub a été arrêté avec le signal `143` pendant
cette duplication dans le run `30515417589`. Elle ne vérifiait pas mieux la
réutilisation que les compteurs, le hash de `nlp.c` et les bornes paramétriques
déjà contrôlés dans les cas principaux. Pour remesurer le gain de compilation,
il faut lancer deux workflows identiques avec
`compile_nlp_evaluators=true/false`; cela évite aussi de biaiser les temps par
la mémoire résiduelle d’un autre OCP.

`--madnlp-c-compile` reste accepté par les CLI pour ne pas bloquer une future
intégration, mais le Bioptim épinglé lève explicitement
`NotImplementedError` avant le premier solve. Le runner refuse donc cette
combinaison tôt et les campagnes MadNLP sont interprétées. Il ne faut ni
présenter un cas à zéro RHO comme un échec de MadNLP, ni contourner cette garde
sans validation des dérivées compilées.

Le mode compilé CasADi peut aussi omettre `Solution.constraints`. L’audit
Cocofest reconstruit désormais \(g(x)\) depuis le NLP symbolique et le vecteur
de décision, puis le compare aux bornes originales \(l_g,u_g\). Ainsi, une
solution compilée n’est plus rejetée seulement parce que `inf_pr` et le champ
de contraintes ne sont pas exportés.

## 9. Faisabilité et définition de la convergence

Un RHO est valide seulement si :

1. le solveur signale une convergence acceptable;
2. toutes les variables sont finies;
3. les bornes de décision originales sont respectées;
4. les contraintes du NLP respectent le seuil commun;
5. le progrès angulaire a le bon signe et la bonne amplitude;
6. l’erreur terminale absolue respecte sa tolérance.

Le seuil physique commun est

\[
\varepsilon_{\mathrm{phys}}=10^{-5}.
\]

Les tolérances internes peuvent différer :

- IPOPT : typiquement `1e-6`;
- MadNLP : `1e-8` dans l’endurance;
- Fatrop : réglages propres à sa structure dans les diagnostics historiques;
- ACADOS : tolérances de dynamique, stationnarité, inégalité et
  complémentarité.

Comparer uniquement les codes de retour natifs serait incorrect. Une solution
peut avoir un statut de succès et manquer le seuil physique, ou inversement
être très proche d’un point acceptable tout en atteignant une limite
d’itérations.

Un échec après fatigue ne constitue pas une preuve mathématique
d’infaisabilité. Les solveurs sont locaux. Une preuve plus forte demanderait
un problème de restauration ou de minimisation de la violation montrant
qu’aucune stimulation admissible ne peut produire le travail requis.

## 10. Mesure des temps

Le benchmark distingue :

- construction du modèle et du graphe;
- préparation du seed commun;
- génération et compilation C;
- création initiale du solveur;
- premier RHO;
- RHO chauds suivants;
- temps natif du solveur;
- temps mural autour de l’appel complet;
- temps mur-à-mur du processus.

Les mesures principales après construction sont :

\[
t_{\mathrm{hot,med}}
=
\operatorname{médiane}
\{t_r:r\ge2,\ r\ \mathrm{valide}\},
\]

et le P90 chaud, qui expose les queues de latence.

`end_to_end_wall_time_s` reste important pour un déploiement complet, mais ne
doit pas être confondu avec le coût d’un RHO après compilation.

Pour les artefacts Fatrop historiques, le rapport exportait par RHO :

- temps total interne;
- temps Hessien;
- temps Jacobien;
- temps des contraintes;
- détection de structure;
- nombres d’évaluations;
- coûts par itération et par évaluation.

Dans le run `30487321536` :

| Fatrop collocation | Full | Reduced |
|---|---:|---:|
| Itérations moyennes | 64.95 | 74.46 |
| Temps interne total, 100 RHO | 832.53 s | 434.82 s |
| Hessien | 402.84 s | 174.93 s |
| Jacobien | 191.33 s | 69.67 s |
| Détection structure | 217.95 s | 173.92 s |
| Temps total par itération | 0.1282 s | 0.0584 s |
| Fraction Hessien + Jacobien | 71.4 % | 56.3 % |

La formulation réduite effectue davantage d’itérations, mais chaque itération
est environ 2.2 fois moins coûteuse. Le gain vient donc surtout du graphe de
dérivées, pas d’une meilleure convergence itérative.

## 11. Résultats Linux historiques de référence

Configuration du run `30487321536`, antérieur à la règle SX-only et conservé
pour la comparaison historique :

- 100 RHO;
- un cycle par OCP;
- 30 stimulations par muscle;
- couple externe nul;
- objectif quadratique de fatigue;
- cible angulaire absolue;
- seuil physique `1e-5`;
- deux échecs consécutifs autorisés;
- seed commun par formulation.

| Solveur | Mécanique | RHO | Médiane chaude | P90 chaud | Mur-à-mur | Coût fatigue | AUC | Min. \(A/A_\mathrm{scale}\) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| IPOPT/MUMPS | full | 100/100 | 2.250 s | 2.566 s | 304.3 s | 4429.09 | 10.0179 | 0.9059 |
| IPOPT/MUMPS | reduced | 100/100 | 1.245 s | 1.480 s | 223.4 s | 658.27 | 4.8262 | 0.9769 |
| MadNLP/MUMPS | full | 100/100 | 2.669 s | 3.083 s | 357.2 s | 4433.09 | 10.0248 | 0.9074 |
| MadNLP/MUMPS | reduced | 100/100 | **1.144 s** | 1.436 s | **177.5 s** | 653.54 | 4.8134 | 0.9771 |
| MadNLP/PARDISO | full | 100/100 | 2.869 s | 3.361 s | 377.7 s | 4433.09 | 10.0248 | 0.9074 |
| MadNLP/PARDISO | reduced | 100/100 | 1.268 s | 1.516 s | 232.9 s | 653.54 | 4.8134 | 0.9771 |
| Fatrop/collocation | full | 100/100 | 8.282 s | 9.584 s | 880.3 s | 4449.07 | 10.0356 | 0.9070 |
| Fatrop/collocation | reduced | 100/100 | 4.216 s | 4.956 s | 473.1 s | 638.32 | 4.7574 | 0.9776 |

Les solveurs convergent vers des résultats voisins à formulation fixée. La
séparation full/reduced est systématique et ne peut pas être attribuée à un
backend particulier.

### 11.1 Fatigue par muscle avec IPOPT

| Mécanique | Muscle | Coût | AUC | \(A_\mathrm{final}/A_\mathrm{scale}\) |
|---|---|---:|---:|---:|
| full | Biceps | 3381.58 | 5.195 | 0.9059 |
| full | Triceps | 771.65 | 2.531 | 0.9594 |
| full | Delt_ant | 193.32 | 1.388 | 0.9849 |
| full | Delt_post | 82.54 | 0.904 | 0.9924 |
| reduced | Biceps | 246.70 | 1.473 | 0.9769 |
| reduced | Triceps | 141.78 | 1.083 | 0.9824 |
| reduced | Delt_ant | 186.43 | 1.362 | 0.9880 |
| reduced | Delt_post | 83.37 | 0.909 | 0.9923 |

Environ 83 % de l’écart de coût provient du Biceps et 17 % du Triceps. Les
deltoïdes sont presque inchangés.

### 11.2 Choix MUMPS; PARDISO et MA57 archivés

PARDISO/MKL ne bat pas MUMPS dans les deux campagnes Linux observées. Le run
`30511306081` donne l’écart le plus net :

- full : `2.571 s` avec MUMPS contre `3.668 s` avec PARDISO, soit MUMPS
  environ 30 % plus rapide;
- reduced : `1.155 s` contre `1.234 s`, soit environ 6 %;
- mur-à-mur : `350.0 s` contre `466.2 s` en full et `178.0 s` contre
  `186.4 s` en reduced.

Cela ne signifie pas que PARDISO est intrinsèquement inférieur. Le runner
dispose de peu de cœurs, le coût des dérivées est important et la
factorisation ne domine pas nécessairement le temps total.

L’étude IPOPT/MA57 sur macOS a montré :

- mêmes objectifs et mêmes itérations que MUMPS dans les runs appariés;
- remainder IPOPT compatible avec une factorisation MA57 plus rapide;
- temps total souvent dominé par des Hessiennes/Jacobiens multithreadés plus
  variables;
- aucun indice d’une mauvaise bibliothèque HSL.

MUMPS reste la référence portable et le seul backend MadNLP actif. MA57 reste
un résultat IPOPT historique; il n’est pas ajouté à la matrice courante.

### 11.3 ACADOS

ACADOS est beaucoup plus rapide lorsqu’il converge. Dans le run de référence,
le SQP-IRK reduced a une médiane proche de `0.080 s`, mais ne valide qu’un
préfixe de huit RHO avant deux échecs consécutifs. Les variantes full, RTI et
ERK ne sont pas encore robustes sur le problème courant.

La comparaison de coût et de fatigue ACADOS avec les NLP après huit cycles
n’est pas une comparaison d’endurance. Elle doit être reportée séparément
tant que le même préfixe de 100 RHO n’est pas validé.

### 11.4 Run diagnostique `30509397708`

Ce run a testé les nouveaux chemins compilés, mais ne doit pas remplacer le
tableau de référence ci-dessus : plusieurs cas ont été arrêtés avant une
mesure comparable.

| Cas | Observation brute | Interprétation |
|---|---|---|
| Fatrop reduced compilé | 100/100, médiane 4.963 s, mur-à-mur 561.2 s | solveur et réutilisation C fonctionnels |
| Fatrop full | 0 RHO | gap non normalisé incompatible avec la structure Fatrop |
| IPOPT compilé | 2 solves de statut 0, objectifs et itérations appariés à l’interprété | faux rejet de l’audit, car `constraints` et `inf_pr` absents |
| MadNLP compilé | 0 RHO | garde `NotImplementedError` Bioptim, pas une non-convergence numérique |
| MadNLP/MUMPS interprété | 5/5 dans l’ablation, médiane 2.766 s | chemin opérationnel |
| ACADOS, 12 variantes | 0 JSON | seed produit avec contraintes initiales strictes, consommateur configuré sans elles |

Le screen MadNLP/MUMPS MX full a résolu 30/30 NLP, mais son audit physique
global est faux : l’erreur de phase absolue reste proche de `3.94e-3 rad`,
au-dessus de `2.01e-3 rad`. Le cas reduced valide le même audit. Cette
différence constante n’est pas un drift de cycle en cycle — les incréments
suivants sont précis à environ `8e-8 rad` — mais révèle encore un décalage
d’ancrage du premier cycle full. Elle doit rester visible et ne doit pas être
« corrigée » en relâchant simplement la tolérance terminale.

Les corrections associées sont maintenant :

1. autoriser un seed `enforce_start_constraints=True` pour un consommateur
   moins strict configuré à `False`, tout en refusant le sens inverse;
2. réévaluer \(g(x)\) pour l’audit des solutions compilées IPOPT;
3. exécuter MadNLP interprété tant que Bioptim ne valide pas sa compilation;
4. corriger la détection de structure Fatrop full en SX avant toute nouvelle
   ablation de scaling.

Le run a aussi confirmé que la compilation IPOPT coûte cher au premier appel
(environ 400 s sur ce runner) pour un gain chaud modeste. Elle peut améliorer
le temps d’un RHO une fois l’OCP construit, mais son seuil d’amortissement est
nettement supérieur à cinq RHO et possiblement supérieur à 100 RHO selon la
machine. Les temps de construction/compilation et les temps chauds doivent
donc toujours rester séparés.

## 12. Expériences historiques et décisions

### 12.1 RK4 avec Fatrop

Le run historique à 100 RHO avait donné :

| Fatrop historique | Full | Reduced |
|---|---:|---:|
| RK4 médiane | 60.9 s | 16.0 s |
| Collocation médiane | 8.28 s | 4.22 s |

RK4 était donc beaucoup plus lent et atteignait aussi des solutions de fatigue
différentes. Il a été retiré de l’endurance. Les futures sondes Fatrop utilisent
la collocation.

### 12.2 Signe du couple

La puissance mécanique externe est

\[
P_{\mathrm{ext}}=\tau_{\mathrm{ext}}\dot\theta.
\]

Le pédalier tourne dans le sens négatif. Un couple signé négatif fournit donc
une puissance positive et assiste le mouvement; un couple signé positif est
résistif. L’interface `--crank-assistance 0.2` convertit la magnitude en couple
signé négatif.

Le benchmark courant fixe le couple à zéro pour faire apparaître la fatigue
plus tôt qu’avec une assistance. Les anciens essais à `+0.20` ou `+0.22 N.m`
signés étudiaient une résistance et ne doivent pas être mélangés avec le
protocole courant.

### 12.3 Deux cycles par OCP

Deux cycles peuvent aider le solveur à anticiper la mémoire lente des états de
Ding. Mais ils :

- doublent presque la taille du NLP;
- créent une couture interne à contraindre;
- réduisent la fréquence de mise à jour du contrôle;
- compliquent la structure Fatrop;
- ne correspondent pas aux seeds ACADOS actuels.

Le benchmark de performance utilise donc un cycle. Deux cycles restent une
étude de robustesse distincte.

### 12.4 Réduction des contrôles

Le masque expérimental fixe les stimulations inactives à `pd0`. Il ne supprime
pas encore les symboles du graphe CasADi; il ajoute des bornes d’égalité. Les
premiers screens ont réduit certaines queues d’itérations, mais modifié le coût
de 3 à 5 %, au-delà du critère de 1 %.

Le benchmark courant utilise :

```text
pulse_width_active_set = none
```

La réduction des contrôles n’explique donc pas l’écart full/reduced.

### 12.5 Écrans archivés

Les résultats ci-dessous expliquent certaines décisions, mais ne doivent pas
être mélangés avec la référence actuelle à couple nul.

#### Alpaqa

Un écran court avec le même seuil physique `1e-5` a donné :

| Configuration Alpaqa | RHO validés | Infaisabilité finale | Limite |
|---|---:|---:|---:|
| pénalité initiale automatique | 0/1 | `4.09e-4` | 60 s |
| pénalité initiale par défaut | 0/1 | `2.91e-2` | 60 s |

La pénalité automatique améliore fortement le résidu, mais reste environ
41 fois au-dessus du seuil. Un ancien test de 600 secondes avait trouvé un
premier candidat proche de la faisabilité, puis une fenêtre décalée avec une
infaisabilité d’environ `4.57e-2`. L’échec est donc lié au problème
multi-fenêtre et pas uniquement à un budget de 60 secondes trop court.

#### Échec sous résistance signée \(+0.22\ \mathrm{N.m}\)

Dans l’expérience historique résistive :

| Solveur | Préfixe valide | Itérations du dernier RHO valide | Itérations des deux échecs | Infaisabilités |
|---|---:|---:|---:|---:|
| MadNLP | 9 | 123 | 1306, 1323 | 0.01068, 0.01413 |
| IPOPT/MUMPS | 10 | 180 | 889, 837 | 0.01159, 0.01444 |

Les deux solveurs échouent dans la même région après un préfixe faisable, ce
qui est compatible avec une limite de capacité induite par la fatigue. Ce
n’est toutefois pas une preuve d’infaisabilité globale, puisque les deux
méthodes sont locales.

#### Masque expérimental de PW

Sur un ancien écran de six cycles :

| Formulation | PW libres | Médiane/P90 | Coût exécuté | AUC |
|---|---:|---:|---:|---:|
| NLP complet | 240/240 | 7.96/15.32 s | 1.77384 | 0.047784 |
| masque, marge 3 | 94/240 | 7.51/8.65 s | 1.82693 | 0.047422 |
| masque, marge 4 | 110/240 | 6.17/6.21 s | 1.86633 | 0.048075 |

La marge 3 enlève un outlier de 272 itérations, mais modifie le coût d’environ
3 %. La marge 4 est plus rapide dans cette répétition, mais modifie le coût
d’environ 5 %. Aucun masque ne satisfait le critère de qualité de 1 %.

Une réduction structurelle plus pertinente demanderait une base cyclique de
faible dimension, par exemple des splines ou une série de Fourier pour les PW,
suivie d’un polissage dans l’espace nodal complet.

## 13. Parallélisme

Le workflow parallélise surtout les expériences :

- une machine IPOPT;
- une machine MadNLP/MUMPS;
- une machine ACADOS;
- full et reduced séquentiels sur une même machine pour réutiliser
  l’environnement.

À l’intérieur d’un RHO, le gain sur 30 ou 48 cœurs ne peut pas être supposé
linéaire. Les limites principales sont :

- portions séquentielles de la méthode de point intérieur;
- factorisations creuses dont le parallélisme dépend de la matrice;
- coût des dérivées CasADi;
- synchronisation et bande mémoire;
- petits blocs musculaires qui ne saturent pas de nombreux cœurs.

Le workflow donne le nombre de cœurs disponibles à CasADi/Bioptim, mais fixe
les pools BLAS/OpenMP imbriqués à un thread pour éviter la sursouscription.

Sur 48 cœurs, le meilleur rendement immédiat reste de lancer plusieurs
solveurs, formulations ou répétitions en parallèle. Le speedup d’un seul RHO
doit être mesuré, pas extrapolé.

## 14. Organisation GitHub Actions

Le workflow est
[`cycling_solver_benchmark_linux.yml`](../../.github/workflows/cycling_solver_benchmark_linux.yml).

Ses étapes principales sont :

1. `prepare-seed` : construction et certification des seeds full/reduced;
2. `prepare-madnlp-stack` : CasADi/libMad et certification de
   `MumpsSolver`;
3. `prepare-acados-stack` : installation et cache ACADOS;
4. `benchmark` :
   - IPOPT full puis reduced;
   - MadNLP/MUMPS interprété, full puis reduced;
5. `acados-smoke` : full/reduced et options séquentiellement;
6. `report` : agrégation des JSON, CSV, logs et patrons de stimulation.

Chaque cas est téléversé immédiatement après sa fin. Une non-convergence
numérique produit un JSON de benchmark; une erreur d’infrastructure fait
échouer le job.

### 14.1 Lancer 100 RHO

```bash
gh workflow run cycling_solver_benchmark_linux.yml \
  --repo mickaelbegon/cocofest \
  --ref codex/acados-pr-refresh \
  -f runner_label=ubuntu-24.04 \
  -f cycles=100 \
  -f cycles_per_window=1 \
  -f crank_assistance_nm=0.00 \
  -f terminal_wheel_q_slack=0.002 \
  -f compile_nlp_evaluators=true \
  -f solver_max_iterations=2000 \
  -f seed_validation_max_iterations=2000 \
  -f acados_smoke_rhos=100 \
  -f acados_option_rhos=5
```

`compile_nlp_evaluators=true` ne concerne que IPOPT. MadNLP reste interprété
dans le même run.

### 14.2 Campagne graduelle obligatoire

Une modification numérique ou d’infrastructure est validée successivement :

1. `cycles=5`, pour certifier les seeds, le runtime MUMPS, SX et les artefacts;
2. `cycles=30`, seulement si tous les cas attendus du palier 5 sont présents,
   sans erreur d’infrastructure ni avertissement libMad;
3. `cycles=100`, seulement après le même verdict au palier 30.

Pour chaque palier, remplacer `cycles`, `acados_smoke_rhos` et, pour le
diagnostic principal ACADOS, `acados_option_rhos` par l’horizon voulu. Les
non-convergences numériques restent des résultats scientifiques; une absence
de JSON, un graphe non-SX ou une option libMad ignorée fait échouer la CI.

Pour mesurer la compilation, relancer sur le même type de runner avec :

```text
-f compile_nlp_evaluators=false
```

### 14.2 Écrans d’options

```bash
gh workflow run cycling_solver_benchmark_linux.yml \
  --repo mickaelbegon/cocofest \
  --ref codex/acados-pr-refresh \
  -f cycles=screen
```

### 14.3 ACADOS uniquement

```bash
gh workflow run cycling_solver_benchmark_linux.yml \
  --repo mickaelbegon/cocofest \
  --ref codex/acados-pr-refresh \
  -f cycles=acados \
  -f cycles_per_window=1 \
  -f crank_assistance_nm=0.00 \
  -f acados_smoke_rhos=100 \
  -f acados_option_rhos=5
```

## 15. Artefacts

Le rapport combiné contient :

- `benchmark-comparison.md` : synthèse lisible;
- `benchmark-comparison.json` : données et provenance complètes;
- `rho-timings.csv` : temps et statuts par RHO;
- `stimulation-patterns.csv` : PW aux checkpoints, angle et phase physiques;
- `muscle-fatigue.csv` : coût, AUC et capacité par muscle;
- `pulse-width-cycle-variation.csv` : variation des PW entre cycles;
- chaque `result.json` individuel;
- chaque `solver.log`.

Les patrons sont comparés aux cycles 10 et 30. Ils sont aussi interpolés selon
la phase réelle du pédalier afin de séparer une stratégie de stimulation
différente d’un simple décalage cinématique.

## 16. Tests locaux

Tests du rapport et du workflow :

```bash
python -m pytest \
  tests/shard1/test_solver_backends.py \
  tests/shard1/test_biorbd_backend_conversion.py \
  tests/shard1/test_plot_backend.py \
  tests/test_cycling_benchmark_summary.py \
  tests/test_solver_option_screen_summary.py \
  tests/shard1/test_reduced_cycling.py \
  tests/shard1/test_periodic_pulse_width.py::test_stimulation_snapshots_use_one_based_cycles_and_real_crank_phase \
  tests/shard1/test_periodic_pulse_width.py::test_common_primal_threshold_is_independent_of_nlp_solver_tolerance \
  tests/shard1/test_periodic_pulse_width.py::test_github_benchmark_compares_physical_threshold_not_solver_tolerance
```

Tests ciblés de l’orchestration GitHub Actions :

```bash
python -m pytest \
  tests/shard1/test_periodic_pulse_width.py::test_github_acados_runner_uses_reference_and_option_profiles_sequentially
```

Tests ACADOS exécutés par son environnement CI dédié :

```bash
python -m pytest \
  tests/shard1/test_reduced_cycling.py \
  tests/shard1/test_periodic_pulse_width.py::test_acados_example_defaults_to_the_assisted_periodic_profile \
  tests/shard1/test_periodic_pulse_width.py::test_common_target_seed_enables_the_robust_acados_reference_preparation \
  tests/shard1/test_periodic_pulse_width.py::test_codegen_names_normalize_user_tag_for_casadi \
  tests/shard1/test_periodic_pulse_width.py::test_control_homotopy_can_relax_stationarity_without_relaxing_feasibility \
  tests/shard1/test_periodic_pulse_width.py::test_phase_one_maps_reduced_mechanics_without_classifying_them_as_fes \
  tests/shard1/test_periodic_pulse_width.py::test_reduced_theta_target_uses_absolute_cycle_reference_without_drift \
  tests/shard1/test_periodic_pulse_width.py::test_acados_irk_transfer_rollout_uses_scaled_variables_and_stage_data
```

Validation syntaxique des scripts :

```bash
python -m py_compile .github/scripts/summarize_cycling_benchmark.py
bash -n .github/scripts/run_cycling_benchmark_case.sh
```

## 17. Priorités scientifiques et numériques

Ordre recommandé :

1. terminer le seed full par relèvement exact du seed reduced;
2. ajouter les résidus cartésiens \(c(q)\) et \(J(q)\dot q\) au rapport, en
   plus des résidus de projection maintenant disponibles;
3. lier la cadence physique entre deux RHO et définir les bornes/cibles avec
   \(\theta,\omega\), pas `q[2],qdot[2]`;
4. relancer 1 puis 30 RHO full/reduced avec le même seed relevé exactement;
5. vérifier les PW, forces, puissance et états Ding sur une trajectoire
   commune;
6. seulement ensuite interpréter la fatigue et étendre à 100 RHO;
7. évaluer `FEASIBILITY_QP -> SQP`, puis seulement si nécessaire construire
   les deux OCP de restauration/optimalité;
8. tester RTI après convergence répétée du SQP complet;
9. comparer deux workflows dédiés, compilation activée puis désactivée, et
   confirmer dans le cas compilé le hash persistant du source généré;
10. profiler séparément dérivées et factorisation MUMPS avant d’augmenter le
    nombre de threads.

Un surrogate neuronal ne doit être envisagé qu’après profilage. La mécanique
réduite de Fourier est déjà très précise et son noyau isolé est environ 40 fois
plus rapide. Les temps Fatrop montrent que le Hessien, le Jacobien et la
détection de structure sont les cibles dominantes. Un réseau neuronal ne sera
pertinent que s’il :

- remplace une fonction réellement dominante;
- fournit des dérivées première et seconde stables;
- respecte la périodicité et les symétries mécaniques;
- conserve des bornes d’erreur compatibles avec `1e-5`;
- est validé hors échantillon sur les trajectoires de fatigue.

À court terme, la projection analytique, SX, la compilation persistante et la
restauration ACADOS sont moins risquées qu’un surrogate appris.

## 18. Conclusion actuelle

- IPOPT/MUMPS full reste la référence robuste.
- MadNLP/MUMPS reduced donne le meilleur temps parmi les NLP 100/100, mais la
  formulation reduced n’est pas encore comparable physiologiquement à full.
- MadNLP utilise explicitement `MumpsSolver`; toute option ignorée devient une
  erreur de CI.
- PARDISO/MKL n’apporte pas de gain et est archivé.
- Fatrop et RK4 sont archivés; Fatrop full doit d’abord devenir compatible SX
  avant une réintégration.
- SX réduit de 57.5 à 60.5 % la médiane chaude face à MX, à objectifs
  identiques à environ `5e-11` près; la campagne active est donc SX-only.
- ACADOS offre le potentiel sous la seconde, mais sa faisabilité inter-RHO
  doit être restaurée.
- Alpaqa ne fonctionne pas sur cette formulation et reste hors production.
