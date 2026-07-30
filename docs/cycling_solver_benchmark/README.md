# Benchmark des solveurs du RHO de pédalage FES

Ce document est la source de référence pour le benchmark de fatigue du
pédalage FES. Il regroupe le problème scientifique, les formulations
mathématiques, la discrétisation, les stratégies de warm-start, les critères
de validation, l’organisation GitHub Actions et l’interprétation des résultats.

La campagne courante est volontairement **SX-only** et compare IPOPT/MUMPS,
MadNLP/MUMPS, FATROP/collocation et ACADOS, en mécanique full et reduced.
FATROP reduced est certifié 100/100; sa formulation full reste une sonde
négative, bloquée lors de l’identification de la structure des gaps. PARDISO
est archivé, car il n’a apporté aucun gain à MadNLP. Alpaqa reste documenté
comme intégration non fonctionnelle.

Les résultats historiques d’endurance proviennent principalement du
[run Linux 30487321536](https://github.com/mickaelbegon/cocofest/actions/runs/30487321536).
La preuve comparative SX/MX vient du
[run Linux 30475768127](https://github.com/mickaelbegon/cocofest/actions/runs/30475768127).
Ils constituent l’historique antérieur à la campagne séquentielle stricte 5,
30 puis 100 RHO maintenant terminée.

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
- PARDISO et RK4 ne font pas partie de la campagne active. FATROP/collocation
  reduced est certifié sur les paliers 5, 30 puis 100 RHO. FATROP full reste
  diagnostique tant que l’interface n’accepte pas sa structure de collocation.
- MadNLP reste interprété : la compilation C est refusée par l’interface
  Bioptim/CasADi épinglée et n’est pas comptée comme une variante valide.
- La convergence du solveur ne suffit pas : chaque RHO est soumis à un audit
  indépendant de faisabilité physique avec un seuil de `1e-5`.
- La mécanique réduite est plus rapide, mais le grand écart de fatigue observé
  historiquement n’est pas une réduction physiologique démontrée. L’équivalence
  full/reduced sur l’endurance reste à établir, car aucun préfixe full strict
  ne dépasse actuellement un cycle.

## 0. Versions Bioptim et reproductibilité

Le benchmark GitHub Actions n’utilise pas le paquet Conda
`bioptim==3.4.0` mentionné dans les instructions d’installation générales de
Cocofest. Il clone directement le fork
[`mickaelbegon/BiorbdOptim`](https://github.com/mickaelbegon/BiorbdOptim)
et effectue chaque checkout par SHA complet. Le SHA est la référence
reproductible; les noms de branche servent uniquement à documenter la
provenance humaine.

### 0.1 Workflow actif de la campagne stricte

| Composant | Version Bioptim réellement utilisée |
|---|---|
| Construction et certification des seeds | `733e442c7b429e20a67a7cf4c2b69694c54513b3` |
| IPOPT full/reduced | `733e442c7b429e20a67a7cf4c2b69694c54513b3` |
| MadNLP/MUMPS full/reduced | `733e442c7b429e20a67a7cf4c2b69694c54513b3` |
| FATROP/collocation full/reduced | `733e442c7b429e20a67a7cf4c2b69694c54513b3` |
| ACADOS full/reduced et variantes | `733e442c7b429e20a67a7cf4c2b69694c54513b3` |

Ce commit appartient à la branche dédiée
`codex/cocofest-acados-v055-exploration`. Il part exactement de
`3523f1745e315f07761159d7e06bd2d876026704`, utilisé par la campagne publiée,
et rassemble les adaptations communes aux différents solveurs :

- ACADOS `v0.5.5`, sous-module
  `59d93e17d2985fdd73fc58b8a83ed8f83a024171`, contre `v0.5.1` auparavant;
- les paramètres numériques modifiables aux `N+1` nœuds;
- la remise à zéro du solveur, les modes SQP/RTI/feasible-QP et les nouveaux
  diagnostics;
- Anderson et le facteur de relaxation Byrd–Omojokun;
- la sauvegarde de l’état primal-dual, qui n’est volontairement pas utilisée
  pour écraser le rollout/projection primal de Cocofest;
- les bornes de contrôle ordonnées et exprimées dans les coordonnées scalées;
- le scaling du guess terminal ACADOS, vérifié explicitement aux stages 0 et
  N par le test ajouté dans `036b9155`;
- un JSON ACADOS isolé dans chaque dossier de code généré;
- une représentation canonique des paramètres runtime permettant de
  réutiliser la même bibliothèque avec de nouvelles données nodales;
- l’export des contraintes non linéaires avec les variables décisionnelles
  scalées attendues par les fonctions de pénalité Bioptim. L’ancienne voie
  appliquait le scaling une seconde fois et créait des résidus artificiels;
- les contraintes non linéaires ACADOS au `Node.START`, nécessaires pour
  ancrer le contact full sans répéter des égalités redondantes à tous les
  nœuds;
- le scaling du guess terminal ACADOS, qui était auparavant injecté en unités
  physiques alors que les nœuds `0..N-1` étaient correctement scalés;
- la normalisation des contraintes de continuité FATROP en coordonnées
  scalées, tout en conservant le resserrement des bornes physiques et
  l’organisation temporelle des variables.

Les tests FATROP ciblés passent localement (`7/7`). Les tests ACADOS qui
requièrent `acados_template` ont été validés dans les environnements Linux des
trois paliers. Le workflow effectue toujours un checkout par SHA complet; le
nom de branche documente seulement la provenance.

Les résultats stricts actifs de la section 11.1 utilisent ce SHA
`a3499cab16d7605b8efa7255cf89f1af6a7c59c9`. Les résultats de la section 11.2,
produits avec l’ancien SHA
`3523f1745e315f07761159d7e06bd2d876026704`, sont conservés explicitement
comme historique non apparié.

### 0.2 Fatrop historique et patchs Bioptim

L’ancienne campagne Fatrop transformait ce checkout, dans cet ordre :

1. [`bioptim-fatrop-c-compile-plugin-case.patch`](../../.github/patches/bioptim-fatrop-c-compile-plugin-case.patch),
   qui corrige la casse du nom du plugin lors du chargement des évaluateurs C;
2. [`bioptim-fatrop-scaled-gaps.patch`](../../.github/patches/bioptim-fatrop-scaled-gaps.patch),
   qui normalise les gaps, les helpers de collocation et les transitions de
   phase pour préserver le bloc identité exigé par Fatrop.

Le second patch était un port minimal du commit Bioptim
`70c384517af48502e5e1bda6c48beb4c515cb8a1`
(`fix: support scaled constraints with FATROP`, 28 juillet 2026). Ce commit est
maintenant intégré directement à la branche multi-solveurs dédiée, avec les
correctifs plus récents de bornes et d’organisation temporelle.

La version effective des résultats Fatrop historiques doit donc être
identifiée par le triplet :

```text
Bioptim 3523f1745e315f07761159d7e06bd2d876026704
+ bioptim-fatrop-c-compile-plugin-case.patch
+ bioptim-fatrop-scaled-gaps.patch
```

Le run `30559215416` a montré deux comportements distincts. Avec scaling
FATROP `full`, le détecteur automatique refuse la structure de $A$ avant le
premier RHO; ce profil reste donc une ablation négative. Avec compilation C,
le chargement utilisait encore le nom sensible à la casse `Fatrop`; la voie
locale de Cocofest inclut désormais `FatropInterface` dans la normalisation du
nom de plugin. Le benchmark SX/collocation suivant revient au scaling `none`,
qui est le profil des anciens CI convergents, tout en conservant les correctifs
de continuité scalée dans la branche pour les essais dédiés.

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

Pour chaque muscle $m$, le modèle de Ding avec fatigue utilise cinq états :

$$
x_m =
\begin{bmatrix}
C_{N,m} & F_m & A_m & \tau_{1,m} & K_{\mathrm M,m}
\end{bmatrix}^{\mathsf T}.
$$

Ils représentent respectivement le complexe calcium-troponine, la force, la
capacité de production de force et deux paramètres dynamiques qui évoluent avec
la fatigue. Avec quatre muscles,

$$
n_{\mathrm{Ding}} = 4 \times 5 = 20.
$$

La dynamique de force peut être écrite sous la forme :

$$
\dot F_m =
\left[
A_m^\mathrm{eff}(PW_m)
\frac{C_{N,m}}{K_{\mathrm M,m}+C_{N,m}}
-
\frac{F_m}
{\tau_{1,m}+\tau_2\frac{C_{N,m}}{K_{\mathrm M,m}+C_{N,m}}}
\right]
\left(f_{\ell,m} f_{v,m}+f_{\mathrm{passif},m}\right),
$$

Dans cette implémentation, le terme dit « passif » multiplie donc la dynamique
de force; il ne correspond pas à une tension passive positive simplement
ajoutée au couple. Sur le profil mécanique courant, son coefficient est
négatif pendant tout le tour pour le Biceps et les deux deltoïdes, et positif
seulement pour le Triceps. La même loi est maintenant utilisée en full et en
reduced, mais une ablation ON/OFF reste nécessaire avant d’interpréter son
effet comme physiologique.

où l’effet de la largeur d’impulsion est

$$
A_m^\mathrm{eff}(PW_m)
=
A_m\left[
1-\exp\left(-\frac{PW_m-pd0_m}{pdt_m}\right)
\right].
$$

Les trois états lents de fatigue suivent :

$$
\dot A_m
=
-\frac{A_m-A_{\mathrm{scale},m}}{\tau_{\mathrm{fat},m}}
+\alpha_{A,m}F_m,
$$

$$
\dot \tau_{1,m}
=
-\frac{\tau_{1,m}-\tau_{1,\mathrm{rest},m}}{\tau_{\mathrm{fat},m}}
+\alpha_{\tau_1,m}F_m,
$$

$$
\dot K_{\mathrm M,m}
=
-\frac{K_{\mathrm M,m}-K_{\mathrm M,\mathrm{rest},m}}{\tau_{\mathrm{fat},m}}
+\alpha_{K_m,m}F_m.
$$

Dans les paramètres courants, $\alpha_A<0$, tandis que
$\alpha_{\tau_1}>0$ et $\alpha_{K_m}>0$.
La force élevée et répétée fait diminuer la capacité normalisée
$A_m/A_{\mathrm{scale},m}$.

### 1.2 Dynamique calcique périodique

La campagne multi-solveurs utilise `periodic_node` pour les formulations full
et reduced. Le calcium est évalué à chaque nœud à partir d’un historique
stationnaire tronqué aux six stimulations précédentes; la largeur d’impulsion
agit sur le recrutement de force, pas sur la somme calcique. À `30 Hz`, avec
`tauc = 0.011 s`, le facteur de décroissance entre deux stimulations vaut
environ `0.0483`. La queue omise après six impulsions est donc de l’ordre de
`3e-7` relativement au régime infini.

Cette estimation indique que `periodic_node` devrait être presque équivalent
au modèle `standard` en régime stationnaire si celui-ci reçoit le même état
initial et cinq stimulations préfenêtre. Elle ne remplace toutefois pas un
benchmark : aucune ablation OCP `standard` contre `periodic_node` appariée
n’est encore publiée. Comme tous les solveurs emploient actuellement la même
formulation calcique, elle ne peut pas expliquer à elle seule un écart
full/reduced, mais la préparation du seed et la continuité inter-RHO peuvent
encore le faire.

### 1.3 Largeurs d’impulsion

Chaque muscle possède 30 commandes de largeur d’impulsion par cycle :

$$
u_k =
\begin{bmatrix}
PW_{1,k} & PW_{2,k} & PW_{3,k} & PW_{4,k}
\end{bmatrix}^{\mathsf T},
\qquad k=0,\ldots,29.
$$

Les bornes physiques sont

$$
pd0_m \le PW_{m,k} \le 600\ \mu\mathrm{s}.
$$

Pour les paramètres Ding courants,

$$
pd0 \simeq 131.405\ \mu\mathrm{s}.
$$

`pd0` est le vrai zéro de recrutement du modèle. Une commande inactive doit
être fixée à `pd0`, et non à une largeur d’impulsion numérique nulle. Les seeds
chargés sont validés et tronqués explicitement dans cet intervalle; chaque
correction produit un warning.

### 1.4 Objectif de fatigue

L’objectif continu quadratique est

$$
J =
10\,000
\int_0^T
\sum_{m=1}^{4}
\left(
1-\frac{A_m(t)}{A_{\mathrm{scale},m}}
\right)^2
\,dt,
$$

où le poids de fatigue actif vaut $10\,000$. Les autres composantes de coût
sont désactivées dans le benchmark :

$$
w_{\mathrm{force}} =
w_{\mathrm{contrôle}} =
w_{\mathrm{angle\ terminal}} =
w_{\dot q} = 0.
$$

Le solveur minimise donc la fatigue, pas la force, la charge électrique ou la
régularité des stimulations. Cette absence de régularisation autorise plusieurs
patrons localement optimaux et peut déplacer le partage Biceps/Triceps sans
modifier fortement l’objectif.

Deux métriques doivent être distinguées :

1. `executed_fatigue_objective`, qui réévalue le coût quadratique sur les
   cycles réellement exécutés;
2. `fatigue_auc_cycles`, définie par

   $$
   \mathrm{AUC}_{\mathrm{fatigue}}
   =
   \int
   \sum_m
   \left(1-\frac{A_m}{A_{\mathrm{scale},m}}\right)
   d(\mathrm{cycle}).
   $$

Le coût quadratique amplifie les muscles fortement fatigués. Un rapport de
coût de 6 ou 7 ne signifie donc pas nécessairement une AUC 6 ou 7 fois plus
grande.

## 2. Receding horizon

Le terme RHO désigne ici une résolution de l’OCP dans la séquence à horizon
glissant. Pour un cycle par OCP, la résolution $r$ optimise

$$
\mathcal P_r:
\quad
\min_{z_r} f(z_r;p_r)
$$

sous

$$
g(z_r;p_r)=0,
\qquad
\underline g_r \le h(z_r;p_r)\le\overline g_r,
\qquad
\underline z_r\le z_r\le\overline z_r.
$$

Le vecteur $p_r$ regroupe les informations qui changent sans modifier le
graphe symbolique :

- l’état exécuté du cycle précédent;
- la cible angulaire absolue;
- les bornes de continuité et les bornes terminales;
- les données nécessaires au décalage des stimulations;
- les états de fatigue accumulés.

La cible terminale est absolue :

$$
\theta_{\mathrm{cible}}(r)
=
\theta_0+r\Delta\theta,
\qquad
\Delta\theta=-2\pi,
$$

avec

$$
\left|\theta(T)-\theta_{\mathrm{cible}}(r)\right|
\le 0.002\ \mathrm{rad}.
$$

Cette définition empêche un drift de même signe d’un cycle au suivant. Une
cible définie à partir du terminal précédent aurait autorisé une accumulation
de petites erreurs, même si chaque RHO respectait localement sa tolérance.
Après chargement du seed commun, $\theta_0$ est systématiquement recalé sur
le premier état effectivement chargé, même si le seed et le consommateur ont
la même formulation mécanique. Sans ce recalage, le run 5 RHO
`30518532002` a produit cinq statuts IPOPT 0 en full, mais un décalage constant
de `3.94e-3 rad` dès le premier cycle, supérieur au slack absolu. Relâcher la
tolérance aurait masqué une cible pré-seed périmée; la correction conserve le
seuil de `0.002 rad`.

Le warm-start primal s’écrit schématiquement :

$$
z_{r+1}^{(0)}
=
\Pi_{\mathcal B_{r+1}}
\left(
\mathcal S z_r^\star
\right),
$$

où $\mathcal S$ décale la trajectoire d’un cycle et
$\Pi_{\mathcal B_{r+1}}$ la projette dans les nouvelles bornes. Les états de
fatigue ne sont pas remis à leur valeur reposée. Dans la campagne historique,
ils étaient toutefois seulement bornés autour du terminal précédent avec des
slacks allant jusqu’à `5e-3` pour certains états, et la vitesse du pédalier
n’était pas raccordée. Il s’agissait donc d’une continuité relâchée, pas d’une
continuité exacte.

Le profil actif fixe maintenant au terminal précédent la vitesse du pédalier
et les 20 états Ding (`Cn`, `F`, `A`, `Tau1`, `Km`) au premier nœud du RHO
suivant. Les traces fusionnées conservent leur grille habituelle pour le calcul
des AUC, mais le JSON publie séparément les deux côtés de chaque couture dans
`state_boundary_jumps`, avec le saut maximal et le RMS par état. Cette
instrumentation doit être validée au palier 5 avant toute comparaison
physiologique de longue durée.

Le benchmark autorise deux échecs consécutifs afin de distinguer un échec
isolé d’une perte persistante de robustesse. Le `validated_cycles` utilisé
pour l’endurance reste néanmoins le préfixe strict avant le premier RHO
invalide.

## 3. Mécanique complète

La mécanique complète possède trois coordonnées généralisées :

$$
q\in\mathbb R^3,
\qquad
\dot q\in\mathbb R^3.
$$

Avec les 20 états musculaires :

$$
n_x^{\mathrm{full}}=20+3+3=26.
$$

Les équations contraintes sont :

$$
M(q)\ddot q+h(q,\dot q)
=
\tau_{\mathrm{muscles}}(q,\dot q,F)
+\tau_{\mathrm{ext}}
+J_c(q)^{\mathsf T}\lambda,
$$

$$
J_c(q)\ddot q+\dot J_c(q,\dot q)\dot q=0.
$$

La seconde équation impose une accélération de contact nulle. Elle n’impose
pas à elle seule :

$$
c(q)=0,
\qquad
J_c(q)\dot q=0.
$$

Ces deux conditions doivent être vraies au départ. Sinon, une erreur de
position ou de vitesse de contact peut être propagée par une dynamique
parfaitement faisable au niveau accélération.

Dans l’implémentation courante, les contraintes explicites de position et de
vitesse du centre du pédalier ne sont ajoutées au nœud initial que lorsque
`enforce_start_constraints=True`. Le profil IPOPT de référence les active,
mais le profil ACADOS historique les désactive. L’option expérimentale
`--full-contact-constraints-all-nodes` les impose à tous les nœuds de tir pour
mesurer si la stabilisation de la variété holonome corrige le drift ACADOS.
C’est une différence centrale avec la mécanique réduite.

## 4. Mécanique réduite

Les deux contraintes holonomes réduisent les trois coordonnées mécaniques à
un degré de liberté. La formulation réduite utilise l’angle physique non
enroulé du pédalier $\theta$ et sa vitesse $\omega$ :

$$
x_{\mathrm{mécanique}}^{\mathrm{reduced}}
=
\begin{bmatrix}\theta & \omega\end{bmatrix}^{\mathsf T},
\qquad
n_x^{\mathrm{reduced}}=20+2=22.
$$

Elle n’impose pas une vitesse constante :

$$
\dot\theta=\omega.
$$

### 4.1 Construction de la variété

Pour chaque angle $\theta$, trois équations déterminent $q(\theta)$ :

1. position horizontale du centre du pédalier;
2. position verticale du centre du pédalier;
3. orientation du vecteur centre-main correspondant à l’angle physique.

La solution périodique est représentée par

$$
q(\theta)
=
w\,s(\theta)+
\sum_{k=0}^{K}
\left(
a_k\cos(k\theta)+b_k\sin(k\theta)
\right),
$$

où $w\,s(\theta)$ porte l’enroulement non périodique de la coordonnée du
pédalier. Les dérivées sont analytiques :

$$
\dot q = T(\theta)\omega,
\qquad
T(\theta)=\frac{dq}{d\theta},
$$

$$
\ddot q
=
T(\theta)\dot\omega
+\frac{d^2q}{d\theta^2}\omega^2.
$$

### 4.2 Projection dynamique

La projection tangentielle donne

$$
M_{\mathrm{eff}}(\theta)=T^{\mathsf T}M(q(\theta))T,
$$

et

$$
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
$$

Les coefficients périodiques suivants sont tabulés puis ajustés par séries de
Fourier :

- inertie effective;
- gravité projetée;
- terme quadratique en vitesse;
- efficacité mécanique de chaque muscle;
- efficacité du couple externe;
- longueurs musculaires normalisées;
- vitesses musculaires par unité de $\omega$.

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

Un audit indépendant a montré que les résultats full/reduced publiés ne
comparaient pas exactement le même problème. Quatre défauts certains sont
maintenant corrigés dans le code, mais les tableaux historiques ne doivent pas
être réinterprétés rétroactivement :

1. `updating_model()` perdait l’activation de la relation de force passive dans
   le full, tandis que reduced la conservait;
2. `qdot[2]` recevait numériquement la borne de cadence physique `omega`, bien
   que `qdot[2] = (dq_2/dtheta) omega`;
3. `common-full.npz` provenait d’un RHO de certification supplémentaire, donc
   full et reduced ne commençaient pas au même instant;
4. le CSV des stimulations utilisait `q[2]` pour la phase full et `theta` pour
   reduced, tout en exportant une vitesse physique.

Le workflow actif utilise désormais `common-reduced.npz` pour les deux
formulations. Le full est relevé par `q=q(theta)` et
`qdot=T(theta) omega`; son solve de certification reste un audit et ne
remplace plus le seed partagé. Les patrons utilisent toujours
`physical_crank_angle_trace` lorsqu’elle existe.

### 5.1 Erreur de contact du seed full

Dans le seed full commun du run `30487321536` :

- le résidu de position du centre du pédalier atteint `0.102 m`;
- son RMS atteint `0.072 m`;
- le résidu de vitesse planaire atteint `0.0376 m/s`;
- la projection sur la variété réduite corrige jusqu’à `0.575 rad`;
- le résidu de vitesse tangentielle atteint `3.23 rad/s`.

La réduction reconstruit toujours une posture sur $c(q)=0$, tandis que la
formulation complète peut poursuivre la trajectoire hors variété. Les longueurs
musculaires, vitesses de contraction, bras de levier et forces nécessaires
sont alors différents.

Le fichier de profil Fourier enregistre maintenant le SHA-256 du `.bioMod`
source. Tout profil sans hash ou construit depuis un autre modèle est rejeté
et reconstruit, afin qu’un cache géométrique périmé ne puisse pas contaminer
la comparaison full/reduced.

Le profil NLP de référence impose maintenant $c(q_0)=0$ et
$J(q_0)\dot q_0=0$ au début de chaque RHO. La branche Bioptim dédiée permet
désormais à ACADOS d’exporter les mêmes égalités `Node.START`. Les ablations
qui ajoutent $c(q_k)=0$, avec ou sans $J(q_k)\dot q_k=0$, à tous les nœuds
restent uniquement diagnostiques : elles sont mathématiquement redondantes
avec la dynamique contrainte au niveau accélération lorsque l’intégration est
exacte et ont déjà dégradé le rang du QP.

Dans tous les cas, chaque trajectoire full est projetée a posteriori sur la
variété réduite. Le rapport contient l’erreur maximale/RMS de configuration,
le résidu maximal/RMS de vitesse tangentielle et les traces physiques
$\theta,\omega$. Une trajectoire qui dépasse `0.01 rad` ou `0.1 rad/s` est
marquée physiquement invalide, même si le solveur retourne le statut zéro.

Le seed historique a aussi été construit à l’aide d’un modèle d’IK qui contient
des transformations fixes différentes de celles du modèle dynamique. Cette
différence géométrique est une autre source plausible de l’incompatibilité
initiale.

Un second défaut a été corrigé dans la construction du warm-start de
collocation. Radau degré 3 n’utilise pas quatre temps uniformes dans chaque
intervalle, mais

$$
\tau=[0,\ 0.1550510257,\ 0.6449489743,\ 1].
$$

L’IK et ses dérivées sont maintenant évalués sur ces temps physiques, y
compris les temps dupliqués entre le point Radau $\tau=1$ et le nœud de tir
suivant. Le recentrage de $\theta$ utilise également cette grille et applique
à $\omega$ la dérivée temporelle de la correction, au lieu de modifier
l’angle seul.

### 5.2 Angle relatif contre angle physique

Dans la formulation complète, `q[2]` est une rotation articulaire relative.
Dans la formulation réduite, $\theta$ est l’angle physique du vecteur
centre-main. Le long de la variété :

$$
\frac{dq_2}{d\theta}\in[0.591,\ 1.452].
$$

Borner `qdot[2]` et $\omega$ avec le même intervalle numérique ne borne donc
pas la même cadence physique. Pour la plage courante
`\omega in [-9.283185, -3.283185] rad/s`, l’enveloppe relevée est :

| Coordonnée | Enveloppe du relèvement | Ancienne borne full |
|---|---:|---:|
| `qdot[0]` | `[-3.3112, 3.6917]` | `[-10, 10]` |
| `qdot[1]` | `[-4.7001, 4.7001]` | `[-14, 10]` |
| `qdot[2]` | `[-13.4810, -1.9400]` | `[-9.2832, -3.2832]` |

Aux cycles 10 et 30 de l’ancien artefact, le relèvement de l’optimum reduced
violait l’ancienne borne full à 3 nœuds sur 30, jusqu’à `2.247 rad/s`.
L’optimum reduced n’était donc littéralement pas admissible dans le full.

Le code échantillonne maintenant `T(theta)` sur un tour, calcule les quatre
produits de chaque extrême de `T_i` et `omega`, ajoute une marge numérique,
puis élargit les bornes full sans resserrer les deux autres coordonnées. La
contrainte non linéaire porte toujours la vraie borne de cadence physique.

De même, appliquer un couple constant sur
`q[2]` donne en coordonnée physique une efficacité modulée par
$dq_2/d\theta$.

Ce point ne contribue pas par le couple externe dans le run courant, puisque
$\tau_{\mathrm{ext}}=0$, mais il affecte les bornes de vitesse et les
diagnostics de phase.

### 5.3 Effet sur les stimulations et la fatigue

Au cycle 10, la solution full atteint près de `600 µs` pour Biceps et Triceps.
La solution reduced reste autour de `225 µs` et `218 µs`. Dans les seeds
communs, les maxima sont respectivement proches de `570/594 µs` contre
`217/208 µs`.

La non-linéarité

$$
1-\exp\left(-\frac{PW-pd0}{pdt}\right)
$$

rend les pics de PW beaucoup plus importants que ne le suggère une simple
comparaison des moyennes. La différence de recrutement se transforme
directement en force, puis en fatigue de $A$, $\tau_1$ et $K_m$.

### 5.4 Expérience appariée requise

Avant d’interpréter le gain de fatigue comme physiologique, il faut :

1. construire le seed full par relèvement exact
   $q=q(\theta)$, $\dot q=T(\theta)\omega$;
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

Les quatre premières briques sont maintenant implémentées et doivent être
recertifiées par la nouvelle CI :

- le seed reduced est résolu en premier;
- le seed full est initialisé avec le relèvement exact
  $q(\theta),T(\theta)\omega$;
- les bornes mécaniques du seed full sont recadrées sans tronquer ce
  relèvement;
- les contraintes de contact initiales et les bornes de cadence physique sont
  actives pour les NLP;
- l’enveloppe des vitesses généralisées full contient tous les relèvements de
  la plage de cadence reduced;
- les phases, cadences et résidus de variété sont audités dans les coordonnées
  physiques.

La contrainte non linéaire de cadence n’est pas encore injectée dans ACADOS :
son export direct avec les états scalés produit artificiellement un résidu
d’inégalité proche de `38.4` et un échec QP dès la première itération.
ACADOS full utilise provisoirement l’enveloppe `qdot` corrigée et l’audit
physique strict a posteriori. Ce cas doit rester marqué comme une limitation
d’interface, pas comme une preuve de non-convergence du problème physique.

Un smoke test local IPOPT/MUMPS à `0 N.m`, un RHO, a donné :

| Formulation | Coût de fatigue exécuté |
|---|---:|
| reduced | `3.599121` |
| full relevé et contraint | `3.604350` |

L’écart n’est plus que d’environ `0.15 %`, contre un facteur proche de 7 dans
le benchmark non apparié. Ce résultat local est très encourageant mais ne
remplace pas encore le test CI apparié de 30 puis 100 RHO.

Pour le full, la tolérance terminale portée par `q[2]` est réduite par
$\min_\theta |dq_2/d\theta|$, afin qu’elle implique la tolérance physique
absolue demandée sur $\theta$. Une contrainte terminale vectorielle
cross/dot plus directe a été prototypée, mais elle fait actuellement avorter
l’initialisation IPOPT dans la pile Bioptim/CasADi locale; elle reste donc
désactivée et la phase physique terminale demeure vérifiée a posteriori.

## 6. Discrétisation et taille du NLP

Le benchmark NLP utilise $N=30$ intervalles et une collocation Radau de
degré $d=3$. Pour un état de taille $n_x$, le nombre approximatif de
variables d’état stockées est

$$
n_x\left[1+N(d+1)\right].
$$

En ajoutant $4N=120$ commandes :

$$
n_z^{\mathrm{full}}
\approx 26(121)+120=3266,
$$

$$
n_z^{\mathrm{reduced}}
\approx 22(121)+120=2782.
$$

La réduction du nombre de variables est donc d’environ 15 %, insuffisante pour
expliquer seule le gain de temps proche de 2. Le principal gain vient de la
réduction de la complexité des dérivées mécaniques.

Pour chaque intervalle, les équations de collocation ont la forme

$$
X_{k,j}
=
X_k+h\sum_{r=1}^{d}a_{jr}f(X_{k,r},U_k),
$$

$$
X_{k+1}
=
X_k+h\sum_{r=1}^{d}b_r f(X_{k,r},U_k).
$$

Le classement temporel des variables place les blocs
$(X_k,U_k,X_{k,1},\ldots)$ de manière à préserver la structure par étage.
Il est utilisé pour IPOPT et MadNLP dans la campagne active. Les essais
Fatrop historiques utilisaient le même ordre. ACADOS conserve son organisation
native par étage.

## 7. Systèmes KKT et solveurs

Une itération de Newton ou de point intérieur conduit schématiquement au
système KKT :

$$
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
$$

où $H_L$ est le Hessien du Lagrangien, $J_g$ le Jacobien des contraintes
et $D$ la contribution de barrière ou de régularisation.

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

### 7.3 FATROP — diagnostic actif, certification endurance en attente

Fatrop exploite la structure d’OCP pour résoudre les systèmes linéaires par
une factorisation de type Riccati plutôt que comme une matrice KKT générique.
Cette exploitation requiert :

- un ordre temporel cohérent;
- des équations de transition identifiables;
- une structure de gap compatible avec la détection automatique.

Le scaling générique des états modifie le coefficient identité du prochain
état dans les gaps. Sans correction, Fatrop refuse la formulation full avec
le diagnostic « structure of A does not correspond » : le bloc associé à
$x_{k+1}$ n’est plus l’identité attendue par le solveur structuré.

Fatrop relâche aussi les bornes de manière relative. Comme certaines capacités
de fatigue valent plusieurs milliers, un facteur relatif apparemment faible
peut produire un écart absolu supérieur au seuil physique. L’interface serre
les bornes transmises à Fatrop tout en conservant les bornes originales pour
l’audit indépendant.

Les anciens tests ont porté sur le commit Bioptim épinglé et le correctif
minimal de la branche `codex/fatrop-scaling-audit`. Chaque gap

$$
S z_{k+1}-\Phi(Sz_k,u_k)=0
$$

en

$$
z_{k+1}-S^{-1}\Phi(Sz_k,u_k)=0,
$$

de sorte que le Jacobien par rapport à $z_{k+1}$ reste exactement
l’identité. La même transformation est appliquée aux helpers de collocation
et aux transitions de phase séquentielles. Le patch conserve simultanément
le resserrement physique des bornes présent dans notre commit Bioptim plus
récent; la branche complète n’est pas utilisée car elle est en retard de
28 commits sur cette base.

Les 7 tests Bioptim dédiés au patch passent localement, mais cela ne suffit
pas : FATROP full échouait encore avec SX lors de la détection de structure,
alors que reduced SX avait seulement passé un smoke 1/1. Le workflow actif
réexécute donc FATROP/collocation full puis reduced dans chaque palier, sur une
machine dédiée, avec scaling d’état `none` et compilation C. Ses résultats ne
seront intégrés au bilan d’endurance qu’après validation numérique et physique
des deux formulations. Les métriques MX antérieures restent seulement des
diagnostics historiques.

Le run `30563108224` confirme précisément la limite full, même avec scaling
`none` : les lignes `3120:3152` sont détectées comme dépendant d’un état de
l’intervalle précédent et la structure annoncée se termine par
`ng=[...,112,1]`. CasADi refuse donc $A$ avant le premier appel à FATROP. Ce
cas reste dans chaque palier comme contrôle négatif explicite; la CI accepte
uniquement cette signature exacte avec `attempted_windows=0`. Toute autre
erreur full, ou toute erreur reduced, demeure une erreur d’infrastructure.
FATROP reduced doit toujours fournir ses RHO physiques et la preuve d’une
unique bibliothèque C réutilisée.

### 7.4 ACADOS

ACADOS résout une suite de QP :

$$
\min_{\Delta z}
\frac12\Delta z^{\mathsf T}H_{\mathrm{GN}}\Delta z
+g^{\mathsf T}\Delta z
$$

sous la linéarisation des dynamiques et des contraintes. La référence utilise :

- SQP complet;
- Hessien de Gauss-Newton;
- intégration IRK;
- HPIPM;
- code généré une fois et réutilisé.

Le workflow actif compare d’abord `v0.5.5` avec exactement les options de la
référence `v0.5.1`. Les nouvelles options ne sont donc pas confondues avec le
changement de version. L’ordre de l’écran court, sur 5 RHO, est :

1. SQP/IRK de référence, duals remis à zéro;
2. remise à zéro de la mémoire ACADOS/HPIPM avant chaque résolution;
3. hot-start HPIPM niveau 2, premier QP initialisé par l’itéré NLP et
   condensation complète `cond_N=N`;
4. contrôle SQP avec `FIXED_STEP`, sans Anderson;
5. même contrôle avec Anderson, seuil d’activation `0.1`;
6. `SQP_WITH_FEASIBLE_QP`, direction Byrd–Omojokun et facteur de slack
   `1.00001`.

Les garde-fous refusent explicitement :

- Anderson avec une globalisation autre que `FIXED_STEP`;
- les résidus QP étendus avec `SQP_WITH_FEASIBLE_QP`, combinaison refusée par
  ACADOS `v0.5.5`;
- un facteur Byrd–Omojokun non fini ou inférieur à un;
- un premier QP initialisé depuis le NLP avec une condensation partielle.

La boucle RHO n’appelle jamais `solve(..., warm_start=previous_solution)`.
Cette voie Bioptim restaurerait `x/u` après le rollout Cocofest et annulerait
la projection mécanique. Le primal reste donc celui du
shift–rollout–projection Cocofest; les duals `pi/lam` sont traités
explicitement et les slacks primaux `sl/su` ne sont pas restaurés.

Chaque variante possède son propre dossier de code, son JSON, son log et son
artefact. `check_reuse_possible` vérifie la compatibilité de la bibliothèque;
les paramètres runtime, les bornes mobiles et la cible terminale sont
réinjectés après toute remise à zéro.

Lorsque l’un des compteurs ACADOS dépasse 5 RHO, le workflow passe
automatiquement en campagne étendue. Il ne reconstruit alors que la référence
reduced et les candidats ayant passé l’écran physique court
(`FIXED_STEP`, Anderson, RTI et IRK léger), plus les éventuelles variantes full
de contact certifiées. Un manifeste `expected-cases.txt` accompagne
l’artefact. Dans ce mode, la CI exige pour chaque cas :

$$
\texttt{success}=\texttt{physical\_success}=\mathrm{true},
\qquad
N_{\mathrm{validated}}=N_{\mathrm{requested}}.
$$

Un statut solveur seul ne peut donc pas faire passer une campagne 30 ou
100 RHO.

SQP-RTI n’effectue qu’une itération SQP par RHO. Il est potentiellement très
rapide, mais exige une trajectoire nominale déjà proche de la variété faisable.
Ce n’est pas encore le cas après plusieurs cycles.

Le run Linux `30548599804`, ACADOS `v0.5.5`, a rendu cette distinction
indispensable :

| Variante reduced, 5 RHO | Succès physique | Médiane chaude | Coût exécuté |
|---|---:|---:|---:|
| SQP/IRK référence | 5/5 | `0.141 s` | `18.03737` |
| SQP/IRK `FIXED_STEP` | 5/5 | `0.103 s` | `18.03737` |
| SQP/IRK Anderson | 5/5 | `0.096 s` | `18.03737` |
| SQP/IRK léger, 2 pas IRK | 5/5 | `0.057 s` | `18.03744` |
| SQP-RTI/IRK | 5/5 | `0.036 s` | `18.03402` |

La remise à zéro de la mémoire ne change ni la solution ni le temps de façon
utile (`0.140 s`). Le hot-start HPIPM niveau 2 échoue dès le premier RHO.
`BYRD_OMOJOKUN` converge, mais est plus lent (`0.121 s`) que
`FIXED_STEP`/Anderson.

En full, aucun SQP ne valide le premier RHO. Le RTI retourne pourtant cinq
statuts zéro en `0.044 s` médian. Ce résultat est un faux positif solveur :
l’audit trouve jusqu’à `4.40 rad/s` de violation de cadence, `8.34 rad/s` de
résidu tangent et une trajectoire très loin de la variété de contact. Le CSV
exporte donc désormais séparément `solver_success`, `physical_success`,
`success` et `mechanical_audit_passed`. La campagne 30 RHO ne doit jamais
sélectionner une variante sur le seul statut natif ACADOS.

Le compteur `physically_validated_cycles` est un préfixe, pas un verdict
global recopié sur tous les RHO. Il part du préfixe NLP convergé et s’arrête au
premier cycle qui dépasse soit la tolérance angulaire absolue, soit la
tolérance de progression entre deux tours. Une non-convergence tardive ne peut
donc plus annuler rétroactivement des cycles antérieurs certifiés. Le run
`30564583487` a exposé ce cas avec IPOPT full : le premier RHO respecte sa
cible, puis le second échoue avant que les RHO isolés suivants ne
reconvergent.

Deux ablations full sont ajoutées à l’écran suivant :

- SQP/IRK avec position et vitesse de contact à tous les nœuds;
- SQP-RTI/IRK avec les mêmes contraintes.

Le run `30550630318` montre que les deux échouent au premier QP avec
`ACADOS_MINSTEP`. Le SQP expose un résidu d’inégalité `17.13`, alors que le
seed passe l’audit mécanique avant le solve. Imposer simultanément
$c(q_k)=0$ et $J(q_k)\dot q_k=0$ à tous les nœuds est trop redondant avec la
dynamique contrainte au niveau accélération et dégrade vraisemblablement le
rang du QP.

La branche Bioptim dédiée exporte maintenant les contraintes non linéaires
`Node.START`. Le full ACADOS reprend donc l’ancrage minimal de la référence de
Kevin :

$$
c(q_0)=0,
\qquad
J_c(q_0)\dot q_0=0.
$$

La dynamique contrainte impose ensuite l’accélération de contact nulle sans
dupliquer position et vitesse comme égalités à tous les nœuds. Cette
formulation doit encore passer le palier Linux 5 RHO; les variantes all-node
restent des diagnostics négatifs de redondance.

La continuité stricte de la cadence a ensuite révélé une discontinuité du
guess cyclique ACADOS. Après le premier RHO reduced du run `30560155975`, la
vitesse terminale vaut environ `-8.03 rad/s`, tandis que le deuxième nœud
recopié de l’ancien cycle reste près de `-5.55 rad/s`. Le défaut initial
atteint `2.29 rad/s` et HPIPM retourne `ACADOS_MINSTEP`. Le rollout IRK brut
est rejeté car il dépasse les bornes de cadence. L’écran 5 RHO contient donc
des variantes SQP et RTI `phase-one`, qui réparent le guess complet avant la
résolution sans changer les équations, les bornes ni l’objectif de l’OCP.

Le premier écran de cette stratégie (`30561812466`) a révélé un défaut
d’orchestration : le lanceur full transmettait l’alias du script bas niveau
`--transfer-phase-one` au comparateur, qui attend
`--shared-transfer-phase-one`. IPOPT, MadNLP et FATROP full s’arrêtaient donc
avant le montage du NLP. Le lanceur utilise maintenant l’option partagée
correcte.

Le callback de transfert est post-résolution : `cycle_idx=1` suit le RHO 1 et
prépare le RHO 2. La trace `transfer_phase_one` imprimée avant le résumé
`window[0]` ne modifie donc pas le seed initial. Dans le run corrigé
`30563108224`, la phase I réduit bien le défaut FES scaled du guess reduced de
`1.0` à environ `0.218`, mais le QP du RHO suivant retourne encore
`ACADOS_MINSTEP`. Le mécanisme agit au bon endroit; sa projection actuelle
n’est simplement pas assez proche de la variété dynamique et des bornes pour
restaurer la convergence.

Le screen strict `30565853248` et le palier 30 `30570144903` confirment que
`FIXED_STEP`, Anderson, l’IRK léger et SQP-RTI reduced ont exactement le même
préfixe : un RHO physique, puis deux échecs consécutifs. Le problème n’est donc
pas corrigé par une option de globalisation. Une explication prioritaire est
maintenant visible dans les NLP reduced : entre les cycles 1 et 2, le
changement maximal de PW atteint environ `469 µs` au Biceps et `80 µs` au
Triceps, tandis que l’homotopie ACADOS limite actuellement le déplacement à
`10 µs`. Les P95 restent pourtant sous `0.3 µs` sur 30 cycles, car ces grands
écarts correspondent à de rares changements de branche active.

Élargir uniformément le trust region n’est donc pas la bonne conclusion. La
prochaine ablation doit :

1. réaligner le guess de PW par angle physique avant le décalage;
2. détecter les quelques nœuds qui quittent `pd0` ou la borne haute;
3. relâcher seulement ces nœuds jusqu’à la pleine plage physique;
4. conserver `10 µs` sur les autres nœuds;
5. rejeter la restauration si la norme des défauts Ding et mécaniques reste
   au-dessus de `1e-5`.

Une autre référence utile sera deux OCP reduced précompilés : un OCP de
faisabilité sans objectif de fatigue, puis l’OCP nominal. La vitesse native
ACADOS du premier RHO (`≈0.1 s`) laisse une marge suffisante pour cette seconde
résolution si elle évite `ACADOS_MINSTEP`.

La phase I actuelle exige un état par extrémité de tir. Elle est donc
compatible avec la transcription ACADOS, mais refuse explicitement la
collocation directe d’IPOPT, MadNLP et FATROP, qui contient des états internes
supplémentaires. Le run `30563108224` a confirmé cette garde avant le premier
solve MadNLP full. Le benchmark collocation n’active plus cette projection;
une version dédiée devra projeter à la fois les extrémités et les points de
Radau.

Dans le run strict sans cette projection (`30560155975`), MadNLP résout les
RHO 1, 3 et 5, mais échoue aux RHO 2 et 4; son préfixe continu reste donc
limité à un cycle. Ce patron alterné est compatible avec un guess cyclique
défectueux, pas avec une arrivée monotone de la fatigue. Pour ACADOS, la phase
de faisabilité est comptée dans le mur-à-mur, tandis que le temps natif du
solveur reste mesuré séparément.

À l’inverse, les deux solveurs reduced qui ne dépendent pas du même algorithme
convergent sur les cinq RHO stricts :

| Solveur reduced strict | RHO | Médiane chaude | Coût exécuté | Fatigue cumulée |
|---|---:|---:|---:|---:|
| MadNLP/MUMPS | 5/5 | `0.938 s` | `19.2349` | `0.16355` |
| FATROP/collocation compilé | 5/5 | `1.904 s` | `19.2382` | `0.16357` |

L’accord du coût à environ `0.02 %` est rassurant pour la formulation reduced,
mais ne valide pas encore son équivalence au full : IPOPT n’a pas terminé et
le full strict n’a pas de préfixe comparable. Le résumé
`state_boundary_jumps`, calculé mais omis du premier JSON agrégé, est désormais
propagé avec les deux côtés de chaque couture.

Après propagation de ce résumé, le run intermédiaire `30561812466` confirme
également cinq coutures strictes pour IPOPT et FATROP reduced : tous les sauts
de `theta`, `omega` et des 20 états Ding sont exactement nuls à la précision
exportée. Les deux solveurs indépendants donnent respectivement des coûts
`19.86837` et `19.86729` (écart `0.0054 %`) et des AUC de fatigue `0.1635768`
et `0.1635703` (écart `0.0040 %`). Ce contrôle croisé valide la cohérence
numérique du reduced à cinq RHO, mais toujours pas son équivalence mécanique
au full.

L’analyse de ces deux écrans a finalement localisé une erreur en amont dans
l’interface Bioptim–ACADOS : les fonctions de pénalité sont construites avec
les variables décisionnelles scalées, mais l’export des contraintes leur
fournissait les expressions non scalées. La fonction de pénalité appliquait
alors une seconde fois les facteurs de scaling. Cela explique simultanément le
résidu de contact artificiel proche de `0.5` malgré un seed projeté et le
résidu de cadence proche de `38`. Le commit Bioptim
`a3499cab16d7605b8efa7255cf89f1af6a7c59c9` corrige les entrées `x`, `u` et
les états algébriques, et ajoute un test avec scaling non unitaire. La borne
non linéaire de cadence physique est donc réactivée pour ACADOS dans le nouvel
écran 5 RHO; les variantes full ne seront admises à 30 RHO que si l’audit
physique confirme le statut natif.

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

Le nouvel écran `sqp-irk-active-set-guard-reduced` traite séparément une autre
source de `MINSTEP`. La région de confiance PW reste à `±10 µs` sur les nœuds
dont le statut de recrutement ne change pas. Autour d’une transition
circulaire recruté/non-recruté, le nœud de transition et ses voisins immédiats
peuvent s’ouvrir jusqu’à `500 µs`, toujours tronqués aux bornes physiques
`[pd0, 600 µs]`. Cela évite qu’un changement légitime de branche soit bloqué
par la continuation, sans supprimer la stabilisation sur tout le cycle. Les
logs et le JSON publient, muscle par muscle, les transitions détectées et les
nœuds réellement ouverts; le CSV d’écran publie aussi le rayon, la marge et
le seuil. Cette variante reste distincte de la référence : le trust et le
garde sont des bornes du solve final, donc ce sous-problème dépend encore du
patron transféré et n’est pas strictement le même OCP que le NLP physique avec
`[pd0, 600 µs]` partout. Si le garde restaure RHO 2, il devra être utilisé
comme phase de faisabilité, puis toutes les bornes physiques seront rétablies
avant un SQP final d’optimalité. Le garde ne peut pas réparer à lui seul une
trajectoire d’états dynamiquement infaisable.

Le palier 5 RHO
[`30582614882`](https://github.com/mickaelbegon/cocofest/actions/runs/30582614882)
confirme ce diagnostic. Le garde ouvre effectivement 19 nœuds autour des
transitions du Biceps et du Triceps, mais RHO 2 échoue encore avec
`ACADOS_MINSTEP`, un résidu dynamique de `0.483` et un résidu de stationnarité
de `154`. Le rollout IRK transféré prédirait une cadence hors borne de
`5.73 rad/s`; les autres défauts dominants sont `F_Triceps`, `F_Delt_post` et
`F_Biceps`. Les PW ne sont donc pas la cause principale.

Ce même log révèle que l’ancienne homotopie `sqp-irk-two-stage-reduced`
sélectionnait uniquement les clés full `q/qdot`. Les états reduced étant
`theta/omega`, l’expansion reportée restait exactement nulle. La sélection
couvre maintenant `q`, `qdot`, `theta` et `omega`; un test interdit la
régression. Cette correction vise directement la restauration du rollout
d’état avant le SQP d’optimalité.

Le screen Linux
[`30583818938`](https://github.com/mickaelbegon/cocofest/actions/runs/30583818938)
confirme que ce correctif agit effectivement sur la bonne variable. Au RHO 2
reduced, le rollout IRK sort `omega` de sa borne de `5.7308 rad/s` et la
relaxation nécessaire atteint `6.1282 rad/s`. Le premier palier, avec bornes
entièrement relâchées, converge avec

$$
\left(r_\mathrm{stat},r_\mathrm{eq},r_\mathrm{ineq},r_\mathrm{comp}\right)
=\left(5.17\,10^{-5},1.25\,10^{-6},4.80\,10^{-11},2.49\,10^{-6}\right).
$$

Le problème transféré est donc restaurable. C’est le resserrement direct
`0 → 1` qui échoue : le meilleur résidu d’égalité du palier physique reste à
`5.93e-2`. Au RHO 3, le dernier itéré du palier relâché est déjà primalement
faisable (`r_eq=4.37e-5`, `r_comp=4.59e-6`) mais sa stationnarité
(`1.75e-3`) dépassait le seuil commun `1e-4`. Exiger l’optimalité à ce palier
intermédiaire n’a pas de justification : il sert seulement à fournir un point
faisable au sous-problème suivant.

La continuation corrigée utilise donc les fractions
`0, 0.125, ..., 0.875, 1`. Les paliers strictement inférieurs à 1 peuvent
transmettre un itéré dont les résidus d’égalité, d’inégalité et de
complémentarité sont sous `1e-4`, même si sa stationnarité ne l’est pas encore.
Le palier physique final à `1` conserve le critère strict sur les quatre
résidus; cette distinction ne peut donc pas certifier une solution qui ne
satisfait que les bornes relâchées. Avec l’expansion observée, chaque pas
déplace une borne d’au plus environ `0.77 rad/s`, contre plus de `6 rad/s`
avec le saut direct. Cette grille est volontairement prudente pour le prochain
palier; elle pourra ensuite être raccourcie en mesurant le plus grand pas qui
conserve la faisabilité.

Enfin, le guess terminal ACADOS est bien envoyé en coordonnées scalées,
$x_N^{ACADOS}=x_N^{physique}/s_x$, comme les stages précédents. Le commit
Bioptim `733e442c7b429e20a67a7cf4c2b69694c54513b3` ajoute un test qui
espionne explicitement les appels `set(0, "x", ...)` et `set(N, "x", ...)`;
le workflow Linux exécute ce test avant les variantes ACADOS.

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
$(T_\mathrm{MX}-T_\mathrm{SX})/T_\mathrm{MX}$, sur les RHO chauds 2 à 30.
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

$$
x_0,\quad l_x,\quad u_x,\quad l_g,\quad u_g.
$$

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

Le cas IPOPT reduced utilise la compilation persistante quand
`compile_nlp_evaluators=true`. Il certifie la compilation sur le problème
d’endurance réellement mesuré. Le full reste SX interprété : sur le runner
GitHub hébergé, la génération C du NLP full à 26 états et collocation a reçu
deux fois un `SIGTERM 143` avant tout artefact, dont une fois exactement après
cinq minutes malgré un heartbeat toutes les 45 secondes
(`30560155975`, `30563108224`). Ce comportement est une limite de ressources
ou de supervision du runner, pas une non-convergence IPOPT. Une ablation full
compilée reste possible sur un runner plus robuste, mais ne doit plus bloquer
la référence scientifique full interprétée.

Pour chaque run compiled reduced multi-RHO, la CI exige :

```text
observed_solves == attempted_windows
runtime_bounds_changed == true
compiled_source_observation_count == attempted_windows
```

Cela vérifie que les bornes mobiles sont effectivement modifiées sans reconstruire
le graphe. Le tracker conserve également taille, `mtime` et SHA-256 de `nlp.c`;
la CI multi-RHO exige une seule version observée et sa réutilisation. Cette
preuve complète le contrôle d’identité du solveur CasADi sans inclure le coût
du hash à chaque RHO : le contenu n’est relu que si taille ou `mtime` changent.
Chaque processus full/reduced s’exécute en outre dans son propre répertoire
temporaire de codegen. Les noms fixes `nlp.c`/`nlp.so` produits par CasADi ne
peuvent donc pas fuir d’une formulation vers l’autre.

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
Cocofest reconstruit désormais $g(x)$ depuis le NLP symbolique et le vecteur
de décision, puis le compare aux bornes originales $l_g,u_g$. Ainsi, une
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

$$
\varepsilon_{\mathrm{phys}}=10^{-5}.
$$

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

$$
t_{\mathrm{hot,med}}
=
\operatorname{médiane}
\{t_r:r\ge2,\ r\ \mathrm{valide}\},
$$

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

## 11. Résultats Linux

### 11.1 Recertification stricte active, 5, 30 puis 100 RHO

Le palier 5 RHO
[`30565853248`](https://github.com/mickaelbegon/cocofest/actions/runs/30565853248)
est vert. Le palier 30 définitif
[`30570144903`](https://github.com/mickaelbegon/cocofest/actions/runs/30570144903)
est également entièrement vert, agrégation incluse. Ces résultats utilisent
Cocofest `9d1073a501be5688a92748b849e2c3d8c9757394`, Bioptim
`a3499cab16d7605b8efa7255cf89f1af6a7c59c9`, ACADOS
`59d93e17d2985fdd73fc58b8a83ed8f83a024171`, des graphes SX, un couple
externe nul et des coutures exactes pour la cadence et les 20 états Ding.

La distinction entre `RHO résolus` et `préfixe strict` est déterminante :
un RHO qui converge après le premier échec reste un diagnostic isolé, mais ne
prolonge pas une trajectoire MHE physiquement exécutable. Le JSON du run
`30565853248` classait correctement ACADOS full à zéro cycle physique à cause
d’un résidu de vitesse tangentielle de `0.804 rad/s`; le premier Markdown
affichait encore le préfixe NLP de un cycle. Le générateur privilégie désormais
explicitement `physically_validated_cycles`.

| Solveur | Mécanique | RHO résolus/tentés | Préfixe strict | Mur-à-mur | Médiane chaude |
|---|---|---:|---:|---:|---:|
| ACADOS SQP/IRK | full | 1/3 | 0/30 | 22.5 s | — |
| FATROP collocation compilé | full | 0/0 | 0/30 | 31.4 s | — |
| IPOPT/MUMPS interprété | full | 28/30 | 1/30 | 300.4 s | — |
| MadNLP/MUMPS interprété | full | 26/30 | 1/30 | 420.7 s | — |
| ACADOS SQP/IRK | reduced | 1/3 | 1/30 | 15.0 s | — |
| FATROP collocation compilé | reduced | 30/30 | 30/30 | 197.8 s | 1.420 s |
| IPOPT/MUMPS compilé | reduced | 30/30 | 30/30 | 154.5 s | **0.726 s** |
| MadNLP/MUMPS interprété | reduced | 30/30 | 30/30 | **68.0 s** | 0.847 s |

FATROP full est une limitation structurelle connue de l’interface collocation,
pas une non-convergence du problème : sa structure `A` contient des
dépendances hors bande que l’interface CasADi/FATROP refuse avant le premier
solve. IPOPT full atteint au RHO 2 une solution primalement faisable
(`6.86e-6 < 1e-5`) mais termine avec `SOLVER_RET_UNKNOWN`; MadNLP échoue
également au deuxième RHO. Le préfixe conservateur vaut donc un cycle pour les
deux solveurs. Les succès ultérieurs ne doivent pas servir au calcul de fatigue
cumulée. ACADOS full obtient un statut NLP valide au premier RHO, mais son
résidu de vitesse tangentielle atteint `0.804 rad/s`; son préfixe physique
reste donc nul.

Sur les 30 cycles réduits réellement exécutés, les trois NLP donnent :

| Solveur reduced | Coût fatigue | AUC, 4 muscles | Min. $A/A_\mathrm{scale}$ |
|---|---:|---:|---:|
| FATROP | 256.488 | 1.521 | 0.964183 |
| IPOPT | 256.519 | 1.521 | 0.964187 |
| MadNLP | 256.415 | 1.520 | 0.964179 |

L’étendue de coût entre les trois solveurs vaut seulement `0.041 %`. Aux
cycles 10 et 30, les patrons FATROP et MadNLP sont aussi presque identiques à
IPOPT : selon le muscle, les RMSE brutes restent sous `0.163 µs` au cycle 10
et sous `0.103 µs` au cycle 30, avec des corrélations proches de 1. Ces
résultats valident fortement la reproductibilité numérique de la formulation
réduite.

Ils ne valident pas encore son équivalence physiologique à la mécanique full.
Sur le seul premier RHO comparable, l’objectif natif reduced est `0.936 %`
plus faible que l’objectif full avec IPOPT comme avec MadNLP. Cet écart court
est modeste, mais aucun solveur full ne fournit encore un préfixe strict de
30 cycles permettant de comparer la fatigue cumulée. L’ancien rapport de
fatigue full/reduced sur 100 RHO reste donc historique.

Le palier 100
[`30573284484`](https://github.com/mickaelbegon/cocofest/actions/runs/30573284484)
est ensuite entièrement vert sur le même SHA Cocofest `9d1073a`. Le verdict
strict est :

| Solveur | Mécanique | RHO résolus/tentés | Préfixe strict | Mur-à-mur | Médiane chaude |
|---|---|---:|---:|---:|---:|
| ACADOS SQP/IRK | full | 1/3 | 0/100 | 18.0 s | — |
| FATROP collocation compilé | full | 0/0 | 0/100 | 20.9 s | — |
| IPOPT/MUMPS interprété | full | 98/100 | 1/100 | 889.8 s | — |
| MadNLP/MUMPS interprété | full | 94/100 | 1/100 | 830.2 s | — |
| ACADOS SQP/IRK | reduced | 1/3 | 1/100 | 11.0 s | — |
| FATROP collocation compilé | reduced | 100/100 | 100/100 | 247.5 s | 1.284 s |
| IPOPT/MUMPS compilé | reduced | 100/100 | 100/100 | 229.1 s | **0.761 s** |
| MadNLP/MUMPS interprété | reduced | 100/100 | 100/100 | **200.4 s** | 0.889 s |

Les trois solveurs reduced restent presque confondus après 100 cycles :

| Solveur reduced | Coût fatigue | AUC, 4 muscles | Min. $A/A_\mathrm{scale}$ |
|---|---:|---:|---:|
| FATROP | 4343.535 | 9.299 | 0.900361 |
| IPOPT | 4343.502 | 9.301 | 0.900366 |
| MadNLP | 4343.271 | 9.297 | 0.900351 |

L’étendue du coût vaut `0.0061 %`. La fatigue reste dominée par le Biceps :

| Solveur reduced | Muscle | Coût | AUC (cycles) | $A_\mathrm{final}/A_\mathrm{scale}$ |
|---|---|---:|---:|---:|
| FATROP | Biceps | 3790.107 | 5.478 | 0.900361 |
| FATROP | Delt_ant | 204.377 | 1.427 | 0.984375 |
| FATROP | Delt_post | 86.814 | 0.929 | 0.991974 |
| FATROP | Triceps | 262.237 | 1.466 | 0.975962 |
| IPOPT | Biceps | 3789.532 | 5.477 | 0.900366 |
| IPOPT | Delt_ant | 204.772 | 1.429 | 0.984351 |
| IPOPT | Delt_post | 87.080 | 0.930 | 0.991945 |
| IPOPT | Triceps | 262.118 | 1.465 | 0.975963 |
| MadNLP | Biceps | 3790.370 | 5.478 | 0.900351 |
| MadNLP | Delt_ant | 204.147 | 1.427 | 0.984427 |
| MadNLP | Delt_post | 86.390 | 0.926 | 0.992022 |
| MadNLP | Triceps | 262.364 | 1.466 | 0.975958 |

Le coût instantané FATROP reduced passe de `3.72` au RHO 1 à `111.65` au
RHO 100, sans non-convergence. Ce palier n’atteint donc pas encore une
impossibilité causée par la fatigue. À l’inverse, les échecs full arrivent
dès le RHO 2, bien avant une fatigue importante. IPOPT full ne compte que deux
échecs isolés et MadNLP six, mais leurs résultats postérieurs au RHO 2
n’appartiennent plus à un préfixe optimal certifié. Au RHO 100, leurs objectifs
isolés diffèrent fortement (`≈111` pour IPOPT, `≈594` pour MadNLP), ce qui
interdit de les utiliser comme substitut à une comparaison d’endurance full.

L’index du résidu dominant permet maintenant de localiser ce premier échec.
Le full possède 26 états : trois positions, trois vitesses et vingt états
Ding. Avec 30 intervalles et une collocation Radau de degré 3, les défauts de
collocation occupent les 3120 premières composantes,

$$
30\ \text{intervalles}\times 26\ \text{états}\times(3+1)=3120.
$$

Les deux composantes de vitesse du centre de pédalier viennent ensuite, puis
les deux composantes de position. L’index `3122`, qui domine les échecs IPOPT
aux RHO 2 et 14 ainsi que quatre des six échecs MadNLP, est donc la première
contrainte de position du centre au début du nouveau RHO. Le terminal du RHO
précédent fournit le nouveau warm-start; la coordonnée et la vitesse du
pédalier ainsi que les 20 états Ding sont raccordées exactement, tandis que
les deux autres coordonnées et vitesses mécaniques restent libres. Le modèle
full historique ne fermait toutefois le contact qu’au début de chaque OCP. Un
léger drift terminal donne donc au RHO suivant un point initial hors variété,
que le solveur doit corriger tout en satisfaisant les états strictement
raccordés. Il ne s’agit pas d’une incompatibilité mathématique certaine,
puisque les coordonnées redondantes restent libres, mais d’un mauvais
warm-start très mal conditionné. Cela explique pourquoi la défaillance
survient immédiatement et presque au même endroit avec deux solveurs de point
intérieur différents. Le reduced ne rencontre pas cette couture : sa
paramétrisation $q=\Phi(\theta)$ appartient algébriquement à la variété de
contact à tous les nœuds.

Imposer position et vitesse du centre à tous les nœuds n’est pas une bonne
stabilisation de ce modèle : sur le test local apparié de deux RHO, IPOPT a
atteint 1000 itérations aux deux fenêtres, avec des résidus primaux de
`6.41e-2` puis `7.24e-3` et environ `286 s` mur-à-mur. Cette variante ajoute
des contraintes redondantes à une dynamique qui n’est pas formulée comme une
dynamique contrainte et détériore fortement le rang et le conditionnement du
KKT. Fermer seulement le terminal ne suffit pas non plus. Avec position et
vitesse terminales, le premier RHO atteint 1000 itérations et `1.03e-4` de
résidu, même si le second converge ensuite. Avec la position terminale seule,
les deux RHO atteignent 1000 itérations; leurs résidus valent `3.20e-4` et
`1.0e-6`, mais aucun ne reçoit un statut de convergence. Ces deux variantes
dures ne sont donc pas retenues.

La piste suivante agit uniquement sur le warm-start : après le transfert, les
coordonnées mécaniques libres du premier nœud sont projetées sur la variété
réduite, tout en conservant exactement la coordonnée et la vitesse du pédalier
déjà fixées. Elle ne change ni l’OCP, ni les bornes physiques, ni les états
Ding; elle remplace seulement un guess full hors contact par son équivalent
cinématique cohérent. Le JSON rapporte l’amplitude de cette correction à
chaque couture afin qu’une projection anormalement grande soit visible.

L’ablation décisive consiste finalement à expliciter une tolérance spatiale de
`20 µm` sur la seule position du centre au début du full. Elle représente
`0.02 %` du rayon de pédalier de `0.1 m`, soit au pire environ
`2e-4 rad` (`0.011°`) en équivalent angulaire; elle reste donc dix fois plus
serrée que la tolérance terminale angulaire de `0.002 rad`. La vitesse du
centre, l’angle du pédalier, la cadence et les vingt états Ding conservent
leurs contraintes exactes.

Sur le test local apparié de deux RHO, cette bande donne deux statuts 0 avec
IPOPT en 88 puis 85 itérations et deux statuts 0 avec MadNLP/MUMPS en 93 puis
69 itérations. Les deux solveurs retrouvent la même somme d’objectifs,
`7.651371`, et les mêmes objectifs par fenêtre à la précision numérique. Au
premier RHO, l’objectif IPOPT passe seulement de `3.754595` avec l’égalité
historique à `3.754515`, soit `-0.0021 %`. Cela confirme que la bande corrige
un seuil numérique de couture et ne crée pas un nouveau mécanisme de
réduction de fatigue.

Le palier local de cinq RHO passe ensuite `5/5` avec les deux solveurs. IPOPT
obtient une somme des objectifs de `20.102977`, une fatigue exécutée de
`19.466525` et un minimum de capacité de `0.984753`; MadNLP/MUMPS obtient
respectivement `20.102820`, `19.466372` et `0.984753`. L’écart relatif de
`0.00078 %` sur la somme des objectifs confirme que les deux méthodes
retrouvent le même régime. Les temps locaux ont été mesurés avec les deux
processus en concurrence et ne doivent donc pas servir de comparaison de
performance. La CI Linux graduelle 5, 30 puis 100 RHO reste la condition de
certification avant d’interpréter de nouveau le full sur 100 cycles.

Le premier palier Linux
[`30581627394`](https://github.com/mickaelbegon/cocofest/actions/runs/30581627394)
confirme séparément les branches NLP full `5/5`; son job ACADOS a échoué
avant les solves à cause d’un fixture de test incomplet et ne change pas ces
résultats :

| Solveur full, 5 RHO | Coût | Fatigue exécutée | AUC | Min. $A/A_\mathrm{scale}$ | Temps solveur | Médiane chaude |
|---|---:|---:|---:|---:|---:|---:|
| IPOPT/MUMPS | 20.09865 | 19.46230 | 0.166329 | 0.984750 | 21.21 s | 4.18 s |
| MadNLP/MUMPS | 20.10282 | 19.46637 | 0.166365 | 0.984753 | 26.05 s | 3.41 s |

IPOPT reduced, exécuté sur une machine séparée, obtient `19.86837`, soit un
coût inférieur de `1.16 %` au full. MadNLP reduced obtient `19.86386`, soit
`1.20 %` de moins que son full. Les capacités finales sont néanmoins presque
identiques après cinq cycles; l’écart de coût vient surtout d’une répartition
d’effort plus coûteuse au Biceps dans le full. Le temps mur-à-mur reduced
d’IPOPT inclut sa compilation persistante et ne doit pas être comparé au full
interprété; les colonnes solveur et médiane chaude sont les métriques
pertinentes ici.

Pour documenter ce que donnent malgré tout les fenêtres isolées au RHO 100,
le tableau suivant reporte l’état au début du RHO 100, donc après 99
transferts. Il ne constitue pas une comparaison d’endurance certifiée pour les
deux lignes full.

| Solveur | Mécanique | Objectif isolé RHO 100 | Biceps | Delt_ant | Delt_post | Triceps |
|---|---|---:|---:|---:|---:|---:|
| IPOPT | full | 111.497 | 0.90684 | 0.97020 | 0.97570 | 0.97813 |
| IPOPT | reduced | 111.653 | 0.90117 | 0.98432 | 0.99192 | 0.97606 |
| MadNLP | full | 594.246 | 0.87438 | 0.84052 | 0.87531 | 0.97448 |
| MadNLP | reduced | 111.649 | 0.90115 | 0.98440 | 0.99200 | 0.97606 |

Les quatre dernières colonnes sont les rapports
$A/A_\mathrm{scale}$. Sur la trajectoire diagnostique IPOPT, le coût isolé du
RHO 100 reste très proche entre full et reduced (`-0.14 %`), mais la
répartition de fatigue diffère : le full conserve environ `0.57` point de
capacité supplémentaire au Biceps, tout en perdant environ `1.41` et `1.62`
points aux deltoïdes antérieur et postérieur. Ces écarts sont utiles pour
orienter l’audit mécanique, mais ils ne deviennent interprétables
physiologiquement qu’après 100 coutures full convergées. La ligne MadNLP full
a déjà bifurqué après six fenêtres non convergées; son coût `5.32` fois plus
élevé que le reduced montre précisément pourquoi une suite de solutions
isolées ne doit pas être présentée comme une endurance.

La compilation reduced est bien persistante : IPOPT et FATROP construisent
chacun une bibliothèque, observent le même source aux 100 solves et changent
les bornes mobiles sans reconstruire le graphe. IPOPT donne le meilleur temps
chaud; MadNLP, malgré son chemin interprété, conserve le meilleur mur-à-mur
reduced sur ce runner.

### 11.2 Campagne SX-only historique, 100 RHO

Le run [`30522170340`](https://github.com/mickaelbegon/cocofest/actions/runs/30522170340)
est la référence historique courante. Ses coûts et fatigues full/reduced ne
doivent pas être interprétés physiologiquement : les seeds n’étaient pas
appariés, la vitesse n’était pas raccordée entre RHO et les états Ding
utilisaient des slacks de couture. Il a été lancé après deux paliers verts à 5 et
30 RHO, avec :

- Cocofest `aac9ff5c2ccec2f16adb6fb1f46932d44e15b7f7`;
- Bioptim `3523f1745e315f07761159d7e06bd2d876026704`;
- libMad `5529f23a6bff33c566ad954da38d352f1f172356`;
- ACADOS `48e223e85f0408ebfd1d8c6d6fb0589e9c41b3aa`;
- un cycle et 30 stimulations par OCP;
- 100 RHO successifs;
- couple externe nul;
- objectif de fatigue seul;
- cible angulaire absolue recalée après chargement du seed, slack
  `0.002 rad`;
- graphes SX uniquement;
- IPOPT compilé une fois par formulation;
- MadNLP interprété avec `MumpsSolver`;
- deux échecs consécutifs permis.

| Solveur | Mécanique | Préfixe strict | Convergés et faisables isolément/tentés | Médiane chaude | P90 chaud | Mur-à-mur | Coût fatigue | AUC | Min. $A/A_\mathrm{scale}$ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| IPOPT compilé/MUMPS | full | 100/100 | 100/100 | **1.444 s** | 1.698 s | 612.4 s | 11406.60 | 16.489 | 0.86736 |
| MadNLP/MUMPS | full | 100/100 | 100/100 | 2.600 s | 2.899 s | **350.1 s** | 11344.72 | 16.460 | 0.86795 |
| ACADOS SQP-IRK | full | 0/100 | 0/2 | — | — | 19.7 s | — | — | — |
| IPOPT compilé/MUMPS | reduced | 100/100 | 100/100 | **1.122 s** | 1.302 s | 255.6 s | 668.10 | 4.849 | 0.97649 |
| MadNLP/MUMPS | reduced | 100/100 | 100/100 | 1.180 s | 1.425 s | **178.4 s** | 652.99 | 4.812 | 0.97707 |
| ACADOS SQP-IRK | reduced | 13/100 | 18/22 | **0.102 s** | 0.104 s | 25.6 s | 46.33¹ | 0.400¹ | 0.98509¹ |

¹ Les métriques ACADOS sont calculées sur le préfixe strict, pas sur
100 cycles. Après le premier échec au RHO 14, les fenêtres 15–18 et 20
convergent et sont faisables lorsqu'elles sont évaluées isolément. Elles ne
prolongent toutefois pas une chaîne MHE valide : le préfixe d'endurance reste
donc limité à 13 cycles. Les échecs consécutifs 21–22 mettent fin à la
campagne.

Les quatre cas NLP terminent donc 100/100 avec succès solveur et physique.
L’erreur angulaire absolue maximale vaut environ `1.470 mrad` en full et
`2.004–2.006 mrad` en reduced; le seuil physique inclut la petite tolérance
numérique autour du slack de `2 mrad`. Entre la fin du premier cycle et celle
du centième, l’écart supplémentaire reste inférieur à `20 µrad` dans les
quatre cas NLP : la dérive résiduelle est négligeable mais pas déclarée
strictement nulle.

IPOPT est le plus rapide par RHO, mais pas mur-à-mur sur 100 RHO. Ses temps
résiduels non attribués aux solves ni à la préparation valent `443.5 s` en
full et `123.1 s` en reduced. Ce résidu n'est pas une mesure instrumentée du
temps de compilation; il est vraisemblablement dominé par la génération et la
compilation initiales, mais peut inclure d'autres frais Python et système.
MadNLP reste interprété et termine respectivement en `350.1 s` et `178.4 s`.
En extrapolant seulement les médianes chaudes, sans supposer que la fatigue
change leur profil, IPOPT compilé rattraperait le temps mur-à-mur de MadNLP
vers environ 330 RHO en full et 1580 RHO en reduced. Ce calcul compare deux
solveurs différents; il ne mesure pas l'amortissement de la compilation
IPOPT par rapport à IPOPT interprété. Cette dernière question exige un cas
IPOPT identique avec `compile=false`.

La preuve de compilation est néanmoins complète : pour chacun des deux cas
IPOPT, `compiled_library_build_count == 1`,
`compiled_source_observation_count == 100`, une seule version de `nlp.c` est
observée et les bornes mobiles changent sans reconstruction du graphe.

#### Fatigue des quatre muscles

| Solveur/mécanique | Muscle | Coût | AUC (cycles) | $A_\mathrm{final}/A_\mathrm{scale}$ |
|---|---|---:|---:|---:|
| IPOPT/full | Biceps | 7161.04 | 7.656 | 0.86736 |
| IPOPT/full | Delt_ant | 918.74 | 2.792 | 0.95191 |
| IPOPT/full | Delt_post | 3225.42 | 5.127 | 0.90968 |
| IPOPT/full | Triceps | 101.41 | 0.914 | 0.98503 |
| MadNLP/full | Biceps | 7162.09 | 7.658 | 0.86795 |
| MadNLP/full | Delt_ant | 916.45 | 2.794 | 0.95222 |
| MadNLP/full | Delt_post | 3164.87 | 5.094 | 0.91067 |
| MadNLP/full | Triceps | 101.30 | 0.914 | 0.98509 |
| IPOPT/reduced | Biceps | 256.28 | 1.496 | 0.97649 |
| IPOPT/reduced | Delt_ant | 186.44 | 1.362 | 0.98805 |
| IPOPT/reduced | Delt_post | 83.38 | 0.909 | 0.99234 |
| IPOPT/reduced | Triceps | 142.00 | 1.083 | 0.98236 |
| MadNLP/reduced | Biceps | 241.96 | 1.460 | 0.97707 |
| MadNLP/reduced | Delt_ant | 186.04 | 1.360 | 0.98807 |
| MadNLP/reduced | Delt_post | 83.19 | 0.908 | 0.99237 |
| MadNLP/reduced | Triceps | 141.80 | 1.083 | 0.98235 |

À formulation fixée, IPOPT et MadNLP produisent des fatigues très proches :
écart de coût total de `0.55 %` en full et `2.31 %` en reduced. En revanche,
la mécanique reduced donne une fatigue beaucoup plus faible : l’AUC totale
est environ 3.4 fois moindre et le coût environ 17 fois moindre qu’en full.
Cet écart vient surtout du Biceps et du deltoïde postérieur et confirme que la
réduction mécanique n’est pas encore un substitut physiologiquement équivalent
à la dynamique full.

#### Patrons de stimulation aux cycles 10 et 30

À mécanique full, IPOPT et MadNLP sélectionnent pratiquement le même patron :
MAE sur les 120 PW de `0.157 µs` au cycle 10 et `0.258 µs` au cycle 30
(écarts maximaux `10.2` et `16.7 µs`). Le Biceps et le deltoïde postérieur
portent des impulsions proches de `600 µs`; le deltoïde antérieur reste
presque partout à `pd0`.

En reduced, les deux NLP sont encore quasi identiques au cycle 10
(`0.019 µs` de MAE). Au cycle 30, ils atteignent deux solutions locales
distinctes compatibles avec des coûts proches. Le recrutement du Biceps
diffère : maximum `422.6 µs` avec IPOPT contre `249.9 µs` avec MadNLP; le
Triceps présente aussi un écart maximal de `64.0 µs`. La MAE globale vaut
`4.61 µs` et l’écart maximal `291.2 µs`. Il faut donc comparer les patrons,
pas seulement le coût scalaire.

ACADOS reduced atteint le cycle 10 mais pas le cycle 30. Au cycle 10, son
patron reste davantage à `pd0`; sa MAE face aux NLP reduced vaut environ
`2.38 µs`, avec un écart maximal proche de `88 µs`. Sa vitesse sous la
seconde est réelle, mais elle n’est pas encore associée à une continuation
robuste jusqu’à 30 ou 100 RHO.

### 11.3 Résultats Linux historiques de référence

Configuration du run `30487321536`, antérieur à la règle SX-only et conservé
pour la comparaison historique uniquement. Pour les mêmes raisons de seed,
contact et continuité inter-RHO, les écarts de fatigue full/reduced du tableau
ne quantifient pas l’effet causal de la réduction mécanique :

- 100 RHO;
- un cycle par OCP;
- 30 stimulations par muscle;
- couple externe nul;
- objectif quadratique de fatigue;
- cible angulaire absolue;
- seuil physique `1e-5`;
- deux échecs consécutifs autorisés;
- seed commun par formulation.

| Solveur | Mécanique | RHO | Médiane chaude | P90 chaud | Mur-à-mur | Coût fatigue | AUC | Min. $A/A_\mathrm{scale}$ |
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

### 11.3 Fatigue historique par muscle avec IPOPT

| Mécanique | Muscle | Coût | AUC | $A_\mathrm{final}/A_\mathrm{scale}$ |
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

### 11.4 Choix MUMPS; PARDISO et MA57 archivés

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

### 11.5 ACADOS

ACADOS est beaucoup plus rapide lorsqu’il converge. Dans le run de référence,
le SQP-IRK reduced a une médiane proche de `0.080 s`, mais ne valide qu’un
préfixe de huit RHO avant deux échecs consécutifs. Les variantes full, RTI et
ERK ne sont pas encore robustes sur le problème courant.

La comparaison de coût et de fatigue ACADOS avec les NLP après huit cycles
n’est pas une comparaison d’endurance. Elle doit être reportée séparément
tant que le même préfixe de 100 RHO n’est pas validé.

### 11.6 Run diagnostique `30509397708`

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
2. réévaluer $g(x)$ pour l’audit des solutions compilées IPOPT;
3. exécuter MadNLP interprété tant que Bioptim ne valide pas sa compilation;
4. recaler la référence angulaire absolue après tout chargement de seed,
   y compris sans changement de formulation mécanique;
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

$$
P_{\mathrm{ext}}=\tau_{\mathrm{ext}}\dot\theta.
$$

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

#### Échec sous résistance signée $+0.22\ \mathrm{N.m}$

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
- une machine FATROP/collocation;
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
   - FATROP/collocation compilé, full puis reduced;
5. `acados-smoke` : full/reduced et options séquentiellement;
6. `report` : agrégation des JSON, CSV, logs et patrons de stimulation.

Chaque cas est téléversé immédiatement après sa fin. Une non-convergence
numérique produit un JSON de benchmark; une erreur d’infrastructure fait
échouer le job.

La génération C du premier RHO IPOPT peut rester plusieurs minutes sans écrire
sur la sortie standard. Deux runners du run `30560155975` ont reçu un signal
d’arrêt pendant ce silence, avant tout retour d’IPOPT ou de MUMPS. Le lanceur
émet désormais un heartbeat toutes les 45 secondes pendant chaque cas; ce
message garde le runner actif sans intervenir dans le processus solveur ni
dans ses mesures de temps.

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
  -f acados_option_rhos=100
```

`compile_nlp_evaluators=true` active les évaluateurs C persistants pour IPOPT
reduced et FATROP reduced. IPOPT full et MadNLP restent interprétés; FATROP
full échoue actuellement avant compilation lors de la détection de structure.

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
Le run 30 RHO `30567069442` a révélé qu’un ancien gate ACADOS exigeait à tort
la convergence physique de toutes les variantes sur tout horizon supérieur à
cinq. Le gate vérifie désormais la présence de tous les cas, l’usage de SX,
un `n_windows > 5` réellement transmis et l’absence d’erreur
d’infrastructure sur les références full/reduced; une non-convergence ACADOS
reste dans le rapport sans rendre le workflow rouge.
Le run de vérification `30568985981` a ensuite révélé que la campagne longue
remplaçait le cas full commun par `sqp-irk-contact-position-full`. Cette
ablation n'a pas le même ensemble admissible et ne peut pas remplacer la
référence dans une comparaison multi-solveurs. Les cas
`sqp-irk-reference-full` et `sqp-irk-reference-reduced` sont donc toujours
exécutés et consignés dans `reference-cases.txt`; les contraintes de contact à
tous les nœuds restent une variante ACADOS séparée.

Cette séquence corrigée est certifiée par les runs verts
[`30570144903`](https://github.com/mickaelbegon/cocofest/actions/runs/30570144903)
à 30 RHO et
[`30573284484`](https://github.com/mickaelbegon/cocofest/actions/runs/30573284484)
à 100 RHO, tous deux construits depuis Cocofest `9d1073a` et Bioptim
`a3499cab16d7605b8efa7255cf89f1af6a7c59c9`.

Pour mesurer la compilation, relancer sur le même type de runner avec :

```text
-f compile_nlp_evaluators=false
```

### 14.3 Écrans d’options

```bash
gh workflow run cycling_solver_benchmark_linux.yml \
  --repo mickaelbegon/cocofest \
  --ref codex/acados-pr-refresh \
  -f cycles=screen
```

### 14.4 ACADOS uniquement

```bash
gh workflow run cycling_solver_benchmark_linux.yml \
  --repo mickaelbegon/cocofest \
  --ref codex/acados-pr-refresh \
  -f cycles=acados \
  -f cycles_per_window=1 \
  -f crank_assistance_nm=0.00 \
  -f acados_smoke_rhos=100 \
  -f acados_option_rhos=100
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

1. isoler l’échec full au RHO 2 en comparant, au même point, les défauts de
   collocation, $c(q)$, $J(q)\dot q$, les forces musculaires et le
   conditionnement KKT; les fenêtres full résolues après cet échec ne doivent
   pas être traitées comme une trajectoire d’endurance;
2. rejouer les mêmes PW et les 20 mêmes états Ding dans les deux mécaniques
   sur plusieurs cycles, puis comparer les RHS mécaniques et musculaires. La
   comparaison de fatigue full/reduced reste suspendue tant que le préfixe
   full strict ne dépasse pas un cycle;
3. pour ACADOS reduced, aligner le warm start sur la phase du pédalier et
   relâcher localement les PW uniquement aux nœuds qui changent de branche
   active. Le maximum observé vaut environ $469\,\mu\mathrm{s}$ alors que le
   P95 reste inférieur à $0.3\,\mu\mathrm{s}$;
4. si ce transfert reste insuffisant, précompiler deux OCP compatibles :
   restauration de faisabilité, puis minimisation de la fatigue. RTI ne doit
   être évalué pour la production qu’après convergence répétée du SQP complet;
5. corriger ou contourner la détection de structure FATROP full. FATROP
   reduced est déjà certifié 100/100; RK4 n’est plus une priorité;
6. prolonger la formulation reduced par paliers de 300 puis 1000 RHO pour
   chercher un échec réellement causé par la fatigue. Le palier 100 ne
   l’atteint pas;
7. comparer deux workflows dédiés, compilation activée puis désactivée, et
   confirmer dans le cas compilé le hash persistant du source généré;
8. profiler séparément dérivées et factorisation MUMPS avant d’augmenter le
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

- La campagne graduelle corrigée 5, 30 puis 100 RHO est entièrement verte. Les
  trois solveurs reduced passent les audits NLP, physiques et de couture
  100/100.
- IPOPT/MUMPS reduced compilé donne le meilleur temps chaud, `0.761 s` de
  médiane. MadNLP/MUMPS reduced interprété donne le meilleur mur-à-mur,
  `200.4 s`; FATROP/collocation reduced compilé termine en `247.5 s`, avec une
  médiane de `1.284 s`.
- Les coûts reduced à 100 RHO diffèrent de seulement `0.0061 %`; les AUC,
  capacités finales et patrons de stimulation sont également cohérents. Cette
  concordance entre trois algorithmes indépendants soutient fortement la
  validité numérique de la formulation reduced.
- Le palier 100 ne provoque pas encore d’échec par fatigue. Le Biceps est le
  muscle limitant, avec une capacité finale voisine de `0.9004`.
- La comparaison physiologique full/reduced n’est toujours pas certifiée.
  IPOPT et MadNLP full résolvent respectivement 98 et 94 fenêtres isolées sur
  100, mais échouent dès le RHO 2 : leur préfixe strict reste d’un cycle. Le
  faible écart d’objectif au premier cycle, environ `0.936 %`, est rassurant
  sans suffire à démontrer l’équivalence sur l’endurance.
- MadNLP utilise explicitement `MumpsSolver`; toute option ignorée devient une
  erreur de CI.
- PARDISO/MKL n’apporte pas de gain et est archivé.
- RK4 FATROP reste archivé. FATROP SX/collocation reduced est maintenant
  certifié 100/100; FATROP full reste bloqué avant le solve par la structure
  des gaps de collocation.
- SX réduit de 57.5 à 60.5 % la médiane chaude face à MX, à objectifs
  identiques à environ `5e-11` près; la campagne active est donc SX-only.
- ACADOS 0.5.5 reduced résout le premier RHO en environ `0.1 s`, mais échoue
  au transfert vers le deuxième avec `ACADOS_MINSTEP`. Les variantes SQP,
  RTI, Anderson et phase I ne prolongent pas encore ce préfixe. ACADOS full
  retourne un premier statut NLP positif hors variété tangentielle et ne
  certifie donc aucun cycle physique.
- La continuité de la cadence et des 20 états Ding est désormais mesurée à
  chaque couture et l’angle terminal est ancré absolument. Ces audits ont
  précisément empêché de confondre fenêtres isolées et endurance exécutable.
- Alpaqa ne fonctionne pas sur cette formulation et reste hors production.
