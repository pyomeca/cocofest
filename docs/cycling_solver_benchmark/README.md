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
- MadNLP full reste interprété. MadNLP reduced dispose maintenant d’un chemin
  C expérimental; la CI doit certifier un seul build, sa réutilisation et
  l’équivalence numérique avant qu’un gain de performance soit revendiqué.
- La convergence du solveur ne suffit pas : chaque RHO est soumis à un audit
  indépendant de faisabilité physique avec un seuil de `1e-5`.
- La mécanique réduite est plus rapide, mais le grand écart de fatigue observé
  historiquement n’est pas une réduction physiologique démontrée. L’équivalence
  full/reduced est réouverte après la correction des bornes de cadence aux
  stages Radau; le palier 5 est cohérent et les paliers 30/100 restent requis.

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
| Construction et certification des seeds | `efd59c39777c83f97058f8d6c1ef472f78f9925d` |
| IPOPT full/reduced | `efd59c39777c83f97058f8d6c1ef472f78f9925d` |
| MadNLP/MUMPS full/reduced | `efd59c39777c83f97058f8d6c1ef472f78f9925d` |
| FATROP/collocation full/reduced | `efd59c39777c83f97058f8d6c1ef472f78f9925d` |
| ACADOS full/reduced et variantes | `efd59c39777c83f97058f8d6c1ef472f78f9925d` |

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
- la compilation des oracles MadNLP par l’importeur CasADi, avec le nom de
  plugin minuscule utilisé de façon cohérente à la génération et au chargement;
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

Le screen suivant
[`30586923568`](https://github.com/mickaelbegon/cocofest/actions/runs/30586923568)
montre à la fois le progrès et la limite d’une grille fixe. Les transferts vers
les RHO 4 et 5 atteignent les bornes physiques `λ=1` avec les quatre résidus
sous `1e-4`. Le transfert critique vers le RHO 2 accepte `λ=0`, mais échoue
dès `λ=0.125` avec `r_eq=8.79e-3`; celui vers le RHO 3 progresse jusqu’à
`λ=0.75` avant d’échouer à `λ=0.875`. Les solutions isolées ultérieures ne
réparent évidemment pas le préfixe strict, qui reste limité à un RHO.

La prochaine variante conserve les fractions ci-dessus comme ancres, mais
ajoute un backtracking local. Lorsqu’une cible échoue, l’itéré fautif est
abandonné, le dernier primal accepté est restauré, puis l’intervalle en
`λ` est bisecté avant de retenter la même ancre. Le pas minimal est
`0.001953125 = 0.125/64` et le nombre total de raffinements est limité à 16
par transfert. Le JSON publie la grille réellement tentée, le dernier
`λ` accepté, le nombre de raffinements, les erreurs de rollback et la raison
d’arrêt. Cette stratégie vise la robustesse; son temps de restauration doit
être compté séparément. Sur le screen fixe, il totalisait déjà `9.71 s` pour
quatre transferts, bien au-delà de la cible temps réel, même si les SQP
nominaux réussis restaient entre `0.04` et `0.14 s`.

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

Le SHA Bioptim actif autorise maintenant `--madnlp-c-compile`. Le support a
été vérifié localement sur un NLP CasADi minimal et sur l’OCP pendule Bioptim :
la génération, le chargement par `Importer("nlp.c", "shell")` et la résolution
MadNLP convergent. Le benchmark active d’abord ce chemin uniquement pour
reduced; full reste interprété pour éviter de reproduire les arrêts mémoire
observés lors de la compilation IPOPT full. La CI exige les mêmes preuves de
réutilisation et de bornes mobiles que pour IPOPT/FATROP. Tant que le palier
Linux 5 RHO n’est pas vert, cette compilation doit rester qualifiée
d’expérimentale et aucun gain ne doit être annoncé.

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
3. exécuter historiquement MadNLP interprété; la validation compilée actuelle
   est décrite à la section 8 et ne réécrit pas ce run diagnostique;
4. recaler la référence angulaire absolue après tout chargement de seed,
   y compris sans changement de formulation mécanique;
5. corriger la détection de structure Fatrop full en SX avant toute nouvelle
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
reduced, MadNLP/MUMPS reduced et FATROP reduced. IPOPT full et MadNLP full
restent interprétés; FATROP full échoue actuellement avant compilation lors
de la détection de structure.

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

### 14.3 RHO reduced contre horizon complet

Le mode isolé `full_horizon` n’exécute pas ACADOS. Le RHO reduced de référence
reste en IPOPT/SX; chaque horizon complet est construit en MX et résolu par
MadNLP/MUMPS :

1. un RHO IPOPT/MUMPS reduced à fenêtres d’un cycle produit une trajectoire
   concaténée jusqu’au plafond demandé ou au premier échec; dans ce dernier
   cas, son préfixe strictement validé devient automatiquement le nouveau
   plafond;
2. cette trajectoire est découpée en seeds appariés pour les horizons complets
   `1, 2, 3, 5, 10, 15, 20, 25, 30`, puis `35…60` par pas de 5 et enfin
   `70, 80, …`;
3. une non-convergence a droit à une seconde tentative et reste visible comme
   trou dans le rapport; la première limite mémoire déclenche un raffinement
   cycle par cycle du dernier intervalle;
4. le pic RSS de tout l’arbre du processus solveur est mesuré. En mode `auto`,
   la limite vaut `12.5 GiB` sur une machine de 16 GiB et `97.5 GiB` sur une
   machine de 128 GiB.

Le RHO utilise IPOPT/SX, dont le préfixe reduced 100/100 est certifié;
l’horizon complet utilise MX. `--single-shot` impose une seule résolution de
l’OCP, `--madnlp-linear-solver mumps` est transmis explicitement et
`--ipopt-no-use-sx` verrouille le graphe MX du problème full. Un pont
IPOPT/MX facultatif raffine d’abord le préfixe RHO relevé sur la mécanique
complète, puis warm-starte MadNLP. Le champ `full_horizon_max_cycles` est un
plafond arbitraire, pas une liste de valeurs codée en dur. Chaque taille est
réinitialisée depuis le préfixe correspondant de la trajectoire RHO reduced
concaténée; les solutions full-horizon précédentes ne sont donc pas propagées
et ne peuvent pas biaiser la branche comparée.

```bash
gh workflow run cycling_solver_benchmark_linux.yml \
  --repo mickaelbegon/cocofest \
  --ref codex/acados-pr-refresh \
  -f runner_label=self-hosted-linux-128gb \
  -f cycles=full_horizon \
  -f full_horizon_max_cycles=100 \
  -f full_horizon_memory_limit_gib=auto \
  -f crank_assistance_nm=0.00 \
  -f terminal_wheel_q_slack=0.002 \
  -f solver_max_iterations=2000
```

L’artefact `cycling-full-horizon-*` conserve le seed RHO concaténé, chaque
préfixe, chaque JSON/solution full-horizon, les logs, les pics RSS et le motif
d’arrêt.

### 14.4 Écrans d’options

```bash
gh workflow run cycling_solver_benchmark_linux.yml \
  --repo mickaelbegon/cocofest \
  --ref codex/acados-pr-refresh \
  -f cycles=screen
```

### 14.5 ACADOS uniquement

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

## 19. Réouverture de la comparaison full/reduced

Le run Linux 30 RHO
[`30588507555`](https://github.com/mickaelbegon/cocofest/actions/runs/30588507555)
change le diagnostic précédent. Avec la tolérance de position de contact
`2e-5 m` au début de chaque RHO, IPOPT et MadNLP convergent désormais 30/30
au sens NLP en full comme en reduced. La correction traite donc bien
l'incohérence de couture qui limitait auparavant le préfixe à un cycle. Les
résultats full de ce run ne sont toutefois plus considérés physiquement
valides après la correction de l'audit tous-points décrite ci-dessous.

| Solveur/formulation | RHO stricts | Fatigue exécutée | AUC | Capacité minimale | Médiane solveur |
|---|---:|---:|---:|---:|---:|
| IPOPT full | 30/30 | 200.343 | 1.39088 | 0.97350 | 3.568 s |
| IPOPT reduced | 30/30 | 256.519 | 1.52072 | 0.96419 | 0.598 s |
| MadNLP/MUMPS full | 30/30 | 192.582 | 1.36867 | 0.97441 | 1.953 s |
| MadNLP/MUMPS reduced | 30/30 | 256.415 | 1.52022 | 0.96418 | 0.779 s |
| FATROP reduced | 30/30 | 256.488 | 1.52055 | 0.96418 | 1.511 s |

La concordance IPOPT/MadNLP/FATROP en reduced est excellente. En revanche,
la fatigue full est 22 à 25 % plus faible que la fatigue reduced selon le
solveur. L'écart est concentré dans le Biceps et, secondairement, le Triceps :

| Muscle | IPOPT full | IPOPT reduced | MadNLP full | MadNLP reduced |
|---|---:|---:|---:|---:|
| Biceps | 93.539 | 145.524 | 86.441 | 145.561 |
| Delt_ant | 66.822 | 66.827 | 66.729 | 66.734 |
| Delt_post | 31.496 | 31.431 | 31.427 | 31.362 |
| Triceps | 8.485 | 12.737 | 7.985 | 12.758 |

Ce résultat ne doit pas être interprété comme un gain physiologique de la
formulation full. Le noyau réduit a été revalidé sur 1000 points aléatoires
de la variété de contact : erreur d'accélération médiane
`5.87e-4 rad/s²`, erreur relative P95 `1.97e-5`, erreur maximale des
coefficients force-longueur `2.73e-9`, force-vitesse `5.55e-9` et passifs
`5.17e-10`. Le fit de Fourier n'explique donc pas un écart de cette amplitude.

L'audit montre plutôt une différence d'ensemble admissible pendant la
collocation. La contrainte de cadence physique full est imposée aux nœuds de
tir, tous les quatre points dans le vecteur de collocation de degré 3. Entre
ces nœuds, la solution full atteint `-9.668 rad/s` avec IPOPT et
`-9.620 rad/s` avec MadNLP, soit des dépassements respectifs de `0.385` et
`0.337 rad/s`. Aux nœuds audités, le dépassement n'est que d'environ
`0.0027 rad/s`. Reduced borne directement `omega` sur tous ses points et ne
présente aucun dépassement significatif. Les écarts hors variété restent
petits (`2.33e-4 rad` en configuration et `5.41e-3 rad/s` en vitesse
tangentielle), mais le régime de vitesse intra-intervalle et la branche locale
de stimulation divergent à partir d'environ RHO 9.

Deux corrections sont maintenant actives :

1. le verdict physique utilise la violation maximale sur tous les points de
   collocation, plus seulement le sous-échantillonnage aux nœuds de tir;
2. la contrainte full calcule la cadence physique au début et aux trois stages
   Radau de chaque intervalle, puis séparément au terminal.

Les pseudo-stages créés par Bioptim en tir direct ne sont pas des variables de
décision. Ils sont donc volontairement exclus de cette fonction : IPOPT et
MadNLP/collocation contraignent les stages Radau, tandis qu'ACADOS/RK contraint
les nœuds de tir et le terminal sans introduire de symbole CasADi libre. Les
stages internes de l'intégrateur ACADOS devront être contrôlés séparément par
rollout ou par une contrainte path native.

Le premier test IPOPT local apparié converge avec ces contraintes. Son coût
de fenêtre vaut `3.7194043`, contre `3.7194409` pour reduced dans le run Linux,
soit un écart relatif d'environ `9.8e-6`. Avant la correction des stages, le
coût full était `3.7545491`. Cette quasi-identité au premier RHO soutient
fortement l'hypothèse causale d'un ensemble admissible discret asymétrique;
elle doit maintenant être confirmée à 5, 30 puis 100 RHO.

Les prochaines ablations doivent donc être appariées :

1. imposer la cadence physique aux points de collocation full, ou utiliser une
   transcription sans états de collocation libres;
2. projeter une solution full complète vers reduced et la réoptimiser, puis
   lever la solution reduced vers full;
3. comparer au même point les RHS mécaniques, les coefficients de Hill, les
   forces et le coût intégré;
4. ne certifier l'équivalence physiologique qu'après disparition de l'écart
   de coût à ensemble admissible identique.

FATROP full reste bloqué avant le solveur : la détection automatique trouve
une structure de gaps incompatible avec l'interface (`nu` et `ng` irréguliers
au dernier intervalle). FATROP reduced reste utilisable.

### 19.1 Warm-start ACADOS adaptatif

Le run
[`30588487246`](https://github.com/mickaelbegon/cocofest/actions/runs/30588487246)
montre que la continuation adaptative des bornes est directionnellement utile,
mais encore trop lente et insuffisante pour la production. Reduced atteint
plusieurs fois la fraction physique `lambda=1`, mais RHO 2 reste bloqué près
de `lambda=0.117` avec un résidu dynamique `1.37e-4`, juste au-dessus du seuil
`1e-4`. La restauration consomme `39.09 s` sur quatre transferts.

L'analyse a aussi trouvé que les statistiques ACADOS peuvent conserver un
historique de résidus obsolète après `ACADOS_QP_FAILURE`. Un palier en
`status=4` ne peut donc plus être certifié par cet historique; seuls les
résidus courants sont acceptés. L'historique reste admissible pour un
`status=2` après de vraies itérations SQP. Le prochain écran porte les paliers
de 20 à 40 itérations, sans relâcher le seuil final à `lambda=1`.

RTI reste une phase de polissage éventuelle, pas un solveur autonome : il est
rapide, mais aucune variante RTI n'a produit un préfixe physique au-delà du
premier RHO.

### 19.2 Validation full/reduced corrigée à 30 RHO

Le run Linux
[`30591104965`](https://github.com/mickaelbegon/cocofest/actions/runs/30591104965),
au SHA Cocofest `eb4fb2d3eb93ffaf0213d4a1393895be3ab7d1e1`, confirme
l’effet de la contrainte de cadence à tous les stages Radau. Ce run utilise
encore le SHA Bioptim historique
`a3499cab16d7605b8efa7255cf89f1af6a7c59c9`, enregistré dans chaque JSON;
la matrice du workflow a depuis été alignée sur le SHA commun documenté à la
section 0.

| Solveur/formulation | RHO stricts | Objectif | Fatigue exécutée | AUC | Capacité minimale | Médiane solveur | Mur-à-mur |
|---|---:|---:|---:|---:|---:|---:|---:|
| IPOPT full | 30/30 | 265.314 | 256.893 | 1.52148 | 0.964144 | 5.022 s | 211.7 s |
| IPOPT reduced | 30/30 | 264.927 | 256.519 | 1.52072 | 0.964187 | 0.806 s | 156.2 s |
| MadNLP/MUMPS full | 30/30 | 265.595 | 257.165 | 1.52156 | 0.964100 | 3.183 s | 158.3 s |
| MadNLP/MUMPS reduced | 30/30 | 264.819 | 256.415 | 1.52022 | 0.964179 | 0.894 s | 73.5 s |
| FATROP reduced | 30/30 | 264.895 | 256.488 | 1.52055 | 0.964183 | 1.502 s | 197.8 s |

Les écarts full/reduced sont maintenant petits et reproduits par deux
solveurs indépendants :

| Métrique relative full - reduced | IPOPT | MadNLP/MUMPS |
|---|---:|---:|
| Objectif | +0.1458 % | +0.2929 % |
| Fatigue exécutée | +0.1457 % | +0.2926 % |
| AUC | +0.0501 % | +0.0878 % |
| Capacité minimale, différence absolue | -4.30e-5 | -7.91e-5 |

La décomposition musculaire ne montre plus l’ancien gain massif et artificiel
du Biceps en full :

| Muscle | IPOPT full | IPOPT reduced | MadNLP full | MadNLP reduced |
|---|---:|---:|---:|---:|
| Biceps | 145.892 | 145.524 | 146.334 | 145.561 |
| Delt_ant | 66.819 | 66.827 | 66.733 | 66.734 |
| Delt_post | 31.425 | 31.431 | 31.362 | 31.362 |
| Triceps | 12.756 | 12.737 | 12.736 | 12.758 |

Une bifurcation transitoire de l’ensemble actif reste visible. Avec IPOPT, le
RHO 7 demande 241 itérations en full contre 39 en reduced. Au cycle 10, une
PW du Biceps diffère d’environ `61.3 us` et la phase du pédalier de
`0.0171 rad` au maximum. Au cycle 30, les patrons se sont réalignés : écart
maximal Biceps `0.343 us` avec IPOPT et `0.071 us` avec MadNLP, écart de phase
inférieur à `0.0021 rad`. La petite différence de fatigue cumulée est donc
compatible avec un changement local et temporaire de branche active, amplifié
par le coût quadratique, plutôt qu’avec une erreur systématique de la
dynamique réduite.

Le mécanisme causal principal de l’ancien écart de 22 à 25 % est ainsi
identifié : full autorisait auparavant une cadence intra-intervalle interdite
à reduced. Il reste des différences de discrétisation, la bande de contact
full de `20 um` et les résidus de projection vers la variété réduite. La
cadence projetée dépasse la borne d’environ `0.00375 rad/s`, cohérente avec le
résidu tangent `0.00416 rad/s` et très inférieure à la tolérance d’audit
`0.1 rad/s`; il ne faut pas la confondre avec la valeur exacte de la contrainte
marker-based du NLP.

La formulation reduced conserve un gain important : médiane solveur environ
`6.23x` plus courte pour IPOPT et `3.56x` pour MadNLP. Le palier 100 RHO doit
encore vérifier que l’écart de fatigue reste inférieur à `0.5 %` et qu’aucune
nouvelle bifurcation tardive ne se développe.

### 19.3 Gate commun 5 RHO, compilation MadNLP et audit ACADOS continu

Le run
[`30592226411`](https://github.com/mickaelbegon/cocofest/actions/runs/30592226411)
est le premier gate où tous les jobs utilisent réellement le même SHA Bioptim
`efd59c39777c83f97058f8d6c1ef472f78f9925d`. IPOPT et MadNLP/MUMPS
convergent 5/5 en full et en reduced; FATROP converge 5/5 en reduced, mais sa
formulation full reste bloquée par la structure de collocation. Les résultats
reduced sont quasiment identiques :

| Solveur | Fatigue exécutée | AUC | Capacité minimale |
|---|---:|---:|---:|
| IPOPT reduced | 19.239254 | 0.163577 | 0.984750 |
| MadNLP/MUMPS reduced | 19.234884 | 0.163553 | 0.984753 |
| FATROP reduced | 19.238211 | 0.163570 | 0.984751 |

MadNLP reduced certifie aussi la compilation persistante : une seule
bibliothèque de `41 775 636` octets est générée, son SHA-256 reste identique
pendant les cinq RHO, les cinq vecteurs de bornes sont distincts et aucun
graphe n’est reconstruit. La médiane chaude est `0.864 s`. Le mur-à-mur de
`95.7 s` reste dominé par la génération et la compilation initiales; ce coût
doit être amorti sur les campagnes longues.

Le meilleur warm-start ACADOS actuel est
`sqp-irk-two-stage-reduced`. Il effectue d’abord une restauration par
homotopie des bornes, puis le SQP de fatigue dans le même OCP compilé. Il
converge 5/5, alors que toutes les variantes reduced sans cette restauration
s’arrêtent après le premier RHO. Les solves nominaux sont rapides, médiane
`0.060 s`, mais les quatre restaurations coûtent `9.45 s` au total. Une
variante adaptative essaie maintenant directement les fractions `[0, 1]` et
ne bissecte qu’en cas d’échec, au lieu de payer systématiquement neuf paliers.

Deux différences de transcription empêchent toutefois de comparer directement
son coût de fatigue `18.355570` aux trois références NLP proches de `19.24`.

Premièrement, le calcium de Ding est raide par rapport au pas :

$$
\tau_c = 0.011\ \mathrm{s},
\qquad
\Delta t = \frac{1}{30}\ \mathrm{s}
\approx 3.03\,\tau_c.
$$

Pour le régime périodique testé, la valeur analytique vaut
`Cn = 0.162982158353`. L’IRK ACADOS, quatre étages de Gauss-Legendre et cinq
sous-pas, donne `0.162982158637`. La collocation Radau degré 3 utilisée par
IPOPT et MadNLP donne `0.152573519058`, valeur reproduite exactement en
réévaluant sa transcription. Elle sous-estime donc ce calcium d’environ
`6.39 %`. L’ERK ACADOS donne au contraire `0.232903256` et reste exclu.
Cette différence explique pourquoi ACADOS obtient plus de force avec moins de
PW surtout pour le Biceps et le Triceps; elle impose une validation NLP à
maillage ou degré raffiné.

Le workflow accepte donc `refined_collocation_validation=true` et
`refined_collocation_rhos=N`. Sur les mêmes machines IPOPT et MadNLP, il
exécute après les références un cas reduced Radau degré 5, interprété pour ne
pas confondre le coût initial de compilation avec l’ablation scientifique.
Ces cas sont étiquetés `ipopt-radau5/reduced` et
`madnlp-mumps-radau5/reduced`; ils ne remplacent pas les références degré 3
dans les critères de succès.

Deuxièmement, ACADOS borne `omega` aux nœuds de tir, pas aux étages internes
IRK. Un intervalle accepté présente

$$
\frac{\Delta\theta}{\Delta t}
=
\frac{-0.321984}{1/30}
=
-9.65952\ \mathrm{rad/s},
$$

alors que la borne basse commune est

$$
-2\pi - 3 = -9.28319\ \mathrm{rad/s}.
$$

La vitesse moyenne sort donc de la borne de `0.37634 rad/s`. Par le théorème
de la moyenne, au moins un point interne la viole nécessairement, même si
tous les `omega` exportés sont admissibles. L’audit mécanique calcule
désormais cette vitesse sécante et rejette une trajectoire ACADOS lorsque la
violation dépasse `0.1 rad/s`.

Les runs
[`30593571421`](https://github.com/mickaelbegon/cocofest/actions/runs/30593571421)
et
[`30594236036`](https://github.com/mickaelbegon/cocofest/actions/runs/30594236036)
ont invalidé la première garde envisagée. Réduire la marge nodale de `3.0` à
`2.5 rad/s` rend déjà la seed commune incohérente avec la dynamique
discrète : le premier SQP retourne `ACADOS_MINSTEP`, avec un résidu d’égalité
d’environ `5.96e-2`, avant tout RHO validé. Surtout, resserrer les valeurs
nodales ne constitue pas une contrainte mathématique sur les étages IRK. Cette
ablation est donc documentée mais retirée des campagnes 30/100 RHO.

Ces deux effets agissent en sens opposés sur l’interprétation : l’intégration
du calcium est plus fidèle avec l’IRK ACADOS, mais son ensemble admissible
continu est trop large entre les nœuds. La campagne suivante doit donc :

1. tester le warm-start à homotopie adaptative et une régularisation douce de
   cadence, sans imposer une vitesse constante;
2. réintégrer densément chaque intervalle ACADOS accepté;
3. raffiner la collocation du calcium pour IPOPT et MadNLP;
4. injecter le patron ACADOS dans ce NLP raffiné;
5. comparer seulement ensuite fatigue, coût et optimalité.

Le full ACADOS souffrait d’un autre défaut de transfert. Le nouvel angle et la
nouvelle vitesse du pédalier étaient fixés au premier nœud, mais les vitesses
redondantes du bras conservaient parfois leur ancienne phase; une couture
observée atteignait `0.64 rad/s`, puis HPIPM retournait
`ACADOS_QP_FAILURE`. Le warm-start full projette maintenant position et
vitesse sur la variété de contact en préservant exactement les deux états du
pédalier. Cette correction est effectivement activée dans le JSON du run
`30594236036`. La référence full résout alors les cinq NLP, mais elle est
correctement rejetée par l’audit continu : sa vitesse sécante minimale vaut
`-9.68586 rad/s`, soit une violation de `0.40268 rad/s`. La même anomalie
existe donc en full et en reduced; elle ne provient pas de la réduction
mécanique.

L’homotopie adaptative `[0, 1]` n’est pas plus rapide sur ce gate. À cinq RHO,
elle consomme `25.32 s` de restauration, contre `10.17 s` pour les neuf
fractions fixes. Le dernier transfert requiert de nombreuses bissections très
proches de `lambda=1`; essayer directement la borne finale répète ainsi
plusieurs échecs coûteux. Elle reste une donnée d’ablation, pas le choix par
défaut.

Deux poids de régularisation de `omega`, `0.1` et `1`, sont maintenant évalués.
Cette pénalité ne fixe pas la cadence : elle laisse la trajectoire varier dans
la bande physique, mais décourage les excursions qui créent les dépassements
internes. Le rapport conserve séparément l’objectif de fatigue exécutée, afin
de mesurer explicitement le prix physiologique de cette régularisation.

Enfin, le premier essai Radau degré 5 a exposé une limite de préparation, et
non une non-convergence : la seed degré 3 contient `121` points d’état, contre
`181` pour le degré 5. L’adaptateur interpole désormais uniquement les états
sur la grille raffinée, conserve exactement les deux extrémités et ne lisse
jamais les contrôles discontinus. Le NLP raffiné restaure ensuite ses propres
équations de collocation. Les itérés ACADOS échoués restent exclus de l’audit
du préfixe déjà validé.

Le run 30 RHO
[`30594633810`](https://github.com/mickaelbegon/cocofest/actions/runs/30594633810)
confirme que cette adaptation de seed fonctionne. MadNLP/MUMPS Radau degré 5
converge `5/5`; après le premier solve à 107 itérations, sa médiane chaude vaut
environ `1.99 s`. IPOPT trouve aussi cinq points primalement faisables, mais
le deuxième solve atteint sa limite de 2000 itérations : les cinq nombres
d'itérations sont `1692, 2000, 84, 92, 120`. Le préfixe optimal strict reste
donc `1/5`. Le run
[`30595640543`](https://github.com/mickaelbegon/cocofest/actions/runs/30595640543)
a ensuite porté cette limite à 5000 : le même RHO reste primalement faisable
mais non optimal après `307.9 s`. Le plafond est donc ramené à 2000 pour cet
audit; MadNLP/MUMPS est le backend raffiné pertinent.

Ce même run a révélé une erreur du rapport, pas des solveurs : les traces
physiques de collocation conservaient leurs points internes, mais la
troncature les lisait comme des nœuds de tir. Le rapport affichait alors zéro
cycle physique malgré un audit mécanique positif. Le stride explicite
`degree + 1` est maintenant utilisé pour IPOPT, MadNLP et FATROP; ACADOS garde
un stride égal à un. Les coûts, fatigues et patrons ne sont exportés que sur
le préfixe ainsi certifié.

Le screen ACADOS
[`30594812890`](https://github.com/mickaelbegon/cocofest/actions/runs/30594812890)
montre les limites de la pénalité de cadence. Avec un poids `0.1`, trois RHO
convergent mais la violation de vitesse sécante atteint encore
`0.437 rad/s`. Avec un poids `1`, le premier RHO reduced passe l'audit continu,
puis le transfert échoue. La référence full converge numériquement `30/30`,
mais sa violation sécante reste `0.403 rad/s`. Le problème n'est donc ni
spécifique à reduced, ni corrigé par une faible pénalité.

Le premier échec full avec le poids `1` était toutefois un problème de
warm-start, et non une incompatibilité de la pénalité. Dans le run
[`30595640543`](https://github.com/mickaelbegon/cocofest/actions/runs/30595640543),
le premier RHO atteint 100 SQP avec un défaut dynamique `1.69e-2`, alors que
les cinq RHO suivants convergent en 8 à 16 itérations. Le workflow applique
donc réellement la stratégie à deux OCP :

1. l'OCP full sans pénalité construit et sauvegarde son premier RHO faisable;
2. l'OCP full avec pénalité de cadence charge ce primal sur la même fenêtre;
3. chaque bibliothèque reste compilée une seule fois et ses paramètres
   mobiles sont ensuite mis à jour pendant la boucle RHO.

Le smoke
[`30596285427`](https://github.com/mickaelbegon/cocofest/actions/runs/30596285427)
valide ce chaînage :

| ACADOS full, poids 1 | Valeur |
|---|---:|
| Préfixe NLP et physique | `6/6` |
| Itérations du premier RHO | `19` |
| Médiane chaude murale | `0.490 s` |
| Violation maximale de vitesse sécante | `0.00751 rad/s` |
| Fatigue exécutée, sans recompter la pénalité | `35.7698` |
| AUC de fatigue | `0.251207` |
| Minimum `A/A_scale` | `0.977954` |
| Mur-à-mur du cas | `25.08 s` |

Les résidus dynamiques finaux sont compris entre environ `2e-10` et `6e-8`.
Cette réussite ne doit pas être transposée à reduced : le poids `1` reduced
reste limité à un RHO; l'homotopie adaptative résout six NLP mais conserve une
violation sécante de `0.377 rad/s`. ACADOS full régularisé devient donc le
candidat d'endurance, tandis que les variantes reduced restent des ablations.

La campagne 100 RHO ne répète plus tout l'écran ACADOS déjà réfuté. Elle
conserve six cas : les références full/reduced, l'homotopie adaptative
reduced pour mesurer la robustesse numérique, la régularisation de poids `1`
en full/reduced, puis un full qui donne une chance additionnelle après
`MAXITER` depuis le meilleur itéré réellement stocké. Le retry est limité à
20 SQP, ne relâche aucune contrainte et n'est autorisé que si la meilleure
faisabilité visitée est inférieure à `2.5e-3`. L'ancienne garde absolue de
rollout `0.2`, les options RTI, Anderson, IRK léger, contact redondant,
homotopie fixe et poids `0.1` restent disponibles comme ablations dans les
campagnes courtes et leurs artefacts séparés.

### 19.4 Recertification full/reduced sur 100 RHO

Le run Linux
[`30596787253`](https://github.com/mickaelbegon/cocofest/actions/runs/30596787253)
est la première comparaison moderne où IPOPT réalise les `100/100` RHO à la
fois en full et en reduced, avec :

- un cycle par OCP et 30 stimulations par cycle;
- un couple externe nul, donc aucune assistance masquant l'arrivée de la
  fatigue;
- les mêmes 20 états de Ding, la même force active, la même force passive et
  les mêmes bornes de PW;
- des graphes SX;
- une collocation Radau de degré 3;
- les bornes de cadence imposées aussi aux étages internes;
- une tolérance de couture de `20 µm` sur la position du centre du pédalier;
- une borne terminale absolue sur l'angle, sans accumulation d'une erreur
  relative au cycle précédent.

La différence mécanique est uniquement la suivante. La formulation full
optimise les six états mécaniques

$$
x_\mathrm{mec}^{\mathrm{full}}
=
\begin{bmatrix}
q_1&q_2&q_3&\dot q_1&\dot q_2&\dot q_3
\end{bmatrix}^{\mathsf T},
$$

alors que la formulation reduced optimise

$$
x_\mathrm{mec}^{\mathrm{red}}
=
\begin{bmatrix}
\theta&\omega
\end{bmatrix}^{\mathsf T},
\qquad
q=\Phi(\theta),
\qquad
\dot q=T(\theta)\,\omega.
$$

La dynamique mécanique reduced est obtenue par projection tangentielle,
sans imposer une cadence constante :

$$
T^{\mathsf T}M(\Phi)T\,\dot\omega
=
T^{\mathsf T}
\left(
\tau_\mathrm{muscle}
+\tau_\mathrm{ext}
-h(\Phi,T\omega)
-M(\Phi)\dot T\,\omega
\right).
$$

Les résultats cumulés sont :

| IPOPT/MUMPS, 100 RHO | Full | Reduced | Écart full/reduced |
|---|---:|---:|---:|
| RHO convergés et préfixe physique | `100/100` | `100/100` | — |
| Objectif total | `4490.808678` | `4486.837031` | `+0.08852 %` |
| Fatigue exécutée | `4347.234072` | `4343.389989` | `+0.08850 %` |
| AUC de fatigue, quatre muscles | `9.304433` | `9.301200` | `+0.03476 %` |
| Minimum $A/A_\mathrm{scale}$ | `0.9003621` | `0.9003667` | `-4.66e-6` |
| Itérations médianes | `58` | `36` | `-37.9 %` |
| Médiane chaude murale | `4.314 s` | `0.951 s` | reduced `4.54x` plus rapide |
| Mur-à-mur du cas | `609.5 s` | `248.4 s` | reduced `2.45x` plus rapide |

La compilation C reduced est réalisée une seule fois, puis la même
bibliothèque est réutilisée pendant les 100 RHO. Le rapport vérifie que les
bornes mobiles changent 100 fois sans reconstruction du graphe. Le full reste
interprété parce que sa compilation persistante n'est pas encore disponible
sur cette transcription; la comparaison de temps mur-à-mur inclut donc ce
choix d'implémentation, tandis que la comparaison physiologique n'en dépend
pas.

La fatigue par muscle confirme l'équivalence :

| Muscle | Coût full | Coût reduced | AUC full | AUC reduced | Capacité finale full | Capacité finale reduced |
|---|---:|---:|---:|---:|---:|---:|
| Biceps | `3793.3635` | `3789.4340` | `5.480659` | `5.477162` | `0.900362` | `0.900367` |
| Delt_ant | `204.6198` | `204.7659` | `1.428163` | `1.428637` | `0.984379` | `0.984351` |
| Delt_post | `86.9963` | `87.0756` | `0.929638` | `0.930080` | `0.991952` | `0.991945` |
| Triceps | `262.2544` | `262.1145` | `1.465973` | `1.465321` | `0.976045` | `0.975964` |

Les patrons de stimulation ne sont pas nécessairement identiques
échantillon par échantillon, car le problème admet plusieurs branches actives
presque équivalentes. Au cycle 10, le RMSE full/reduced du Biceps vaut
`12.24 µs` et le maximum `61.38 µs`; au cycle 30 ils tombent à `0.028 µs` et
`0.153 µs`. Le Triceps reste sous `0.31 µs` de RMSE aux deux horizons et les
deux deltoïdes sont pratiquement confondus. Une nouvelle bifurcation de la
branche active du Biceps apparaît au cycle 100, avec un RMSE de `9.30 µs`,
sans produire d'écart significatif de fatigue cumulée. Comparer seulement les
PW brutes serait donc trop sévère; il faut aussi comparer la phase, le couple,
le coût et les états de fatigue.

FATROP reduced converge également `100/100` :

| FATROP reduced | Valeur |
|---|---:|
| Fatigue exécutée | `4343.534710` |
| AUC de fatigue | `9.299235` |
| Minimum $A/A_\mathrm{scale}$ | `0.9003607` |
| Médiane chaude murale | `1.398 s` |
| Mur-à-mur du cas | `327.9 s` |

Son écart de fatigue avec IPOPT reduced est d'environ `0.0033 %`. Il valide
donc indépendamment l'optimum reduced, mais reste environ `1.47x` plus lent
par RHO chaud. FATROP full demeure bloqué avant le solve par la détection de
structure de l'interface CasADi/FATROP; ce résultat n'est ni un échec de
faisabilité ni un argument contre la mécanique full.

MadNLP/MUMPS est plus rapide que le full IPOPT lorsque ses fenêtres
convergent, mais cette campagne révèle une sensibilité au transfert :

- le full converge `99/100`, avec un premier échec au RHO 81
  (`maxiter=2000`, résidu primal `2.46e-2`), puis récupère dès le RHO 82;
- le reduced échoue au RHO 1 avec une seed IPOPT située sur une autre branche,
  puis converge aux RHO 2 à 100;
- l'échec isolé suivi d'une récupération signale une difficulté numérique,
  mais cette récupération seule ne certifie pas la chaîne : le transfert a
  déjà propagé l'état issu du solve non faisable. La convergence d'IPOPT sur
  le même RHO est l'argument qui exclut ici une impossibilité créée par la
  fatigue;
- le préfixe physique strict vaut néanmoins 80 pour le full et 0 pour le
  reduced : les fenêtres postérieures ne sont jamais agrégées comme une
  trajectoire exécutable.

Le rapport corrige désormais le numéro du premier échec même lorsque des RHO
ultérieurs réussissent. La seed commune est immuable à l'intérieur d'un run,
mais le solve IPOPT qui la produit peut sélectionner des branches non convexes
différentes entre deux runs. MadNLP effectue donc maintenant, avant le
chronométrage, un raffinement IPOPT périodique sur sa transcription exacte.
Ce coût est imputé à `initial_guess_preparation_time_s`; il ne reconstruit pas
le graphe pendant la boucle. Cette correction doit passer successivement les
gates 5, 30 et 100 RHO avant que MadNLP soit déclaré robuste.

Le gate 5 RHO
[`30598124791`](https://github.com/mickaelbegon/cocofest/actions/runs/30598124791)
valide la première étape :

| MadNLP/MUMPS avec raffinement IPOPT | Full | Reduced |
|---|---:|---:|
| Préfixe strict | `5/5` | `5/5` |
| Objectif | `19.862934` | `19.863857` |
| Fatigue exécutée | `19.233967` | `19.234884` |
| Préparation initiale | `56.38 s` | `43.84 s` |
| Médiane chaude murale | `3.494 s` | `1.110 s` |

Le coût de préparation augmente volontairement, mais il est payé une fois
avant la séquence et n'entre pas dans la médiane chaude. L'écart de fatigue
full/reduced vaut seulement `0.0048 %` sur ce gate. Les gates 30 et 100 RHO
ci-dessous déterminent ensuite si ce transfert reste robuste avec la fatigue.

Le gate 30 RHO
[`30599827365`](https://github.com/mickaelbegon/cocofest/actions/runs/30599827365)
valide ensuite la robustesse intermédiaire du même transfert :

| Solveur, 30 RHO | IPOPT full | IPOPT reduced | MadNLP full | MadNLP reduced | FATROP reduced |
|---|---:|---:|---:|---:|---:|
| Préfixe solveur et physique | `30/30` | `30/30` | `30/30` | `30/30` | `30/30` |
| Fatigue exécutée | `256.892781` | `256.518977` | `257.165003` | `256.414670` | `256.488100` |
| AUC de fatigue | `1.521481` | `1.520718` | `1.521559` | `1.520224` | `1.520554` |
| Minimum $A/A_\mathrm{scale}$ | `0.9641440` | `0.9641870` | `0.9641002` | `0.9641793` | `0.9641831` |
| Médiane chaude solveur | `5.063 s` | `1.110 s` | `2.925 s` | `0.917 s` | `1.405 s` |
| Mur-à-mur du cas | `212.64 s` | `164.36 s` | `194.18 s` | `174.11 s` | `195.53 s` |

Sur ce gate, la fatigue full dépasse la reduced de `0.146 %` avec IPOPT et
de `0.293 %` avec MadNLP. Les AUC ne diffèrent que de `0.050 %` et `0.088 %`.
La réduction accélère la médiane chaude de `4.56x` avec IPOPT et de `3.19x`
avec MadNLP. FATROP full reste un échec structurel avant le solve; il ne doit
pas être inclus dans ces ratios.

La différence est surtout portée par le Biceps. Avec IPOPT, son coût passe de
`145.5241` en reduced à `145.8920` en full et sa capacité finale de
`0.9641870` à `0.9641440`. Les trois autres muscles restent chacun à moins de
`0.02` unité de coût entre les formulations. MadNLP donne la même conclusion :
le coût Biceps vaut `145.5612` en reduced et `146.3336` en full; les écarts
des deltoïdes et du Triceps sont inférieurs à `0.023`. Cette localisation et
la cohérence avec IPOPT indiquent un petit effet mécanique réel ou de
transcription, pas le large biais physiologique observé avant la correction
des bornes de cadence aux étages de collocation.

Le gate 100 RHO
[`30600487081`](https://github.com/mickaelbegon/cocofest/actions/runs/30600487081)
complète la comparaison d'endurance. Le workflow est vert parce que tous les
cas et artefacts ont été exécutés correctement; cela ne signifie pas que
chaque solveur a convergé sur les 100 RHO :

| Solveur, 100 RHO | IPOPT full | IPOPT reduced | MadNLP full | MadNLP reduced | FATROP reduced |
|---|---:|---:|---:|---:|---:|
| Solves réussis | `100/100` | `100/100` | `99/100` | `99/100` | `100/100` |
| Préfixe physique strict | `100` | `100` | `80` | `98` | `100` |
| Premier RHO en échec | — | — | `81` | `99` | — |
| Fatigue du préfixe | `4347.708565` | `4343.497299` | `2488.699600` | `4128.757100` | `4343.534710` |
| AUC du préfixe | `9.304984` | `9.301330` | `6.543290` | `9.003140` | `9.299235` |
| Minimum $A/A_\mathrm{scale}$ | `0.9003577` | `0.9003658` | `0.9163140` | `0.9019440` | `0.9003607` |
| Médiane chaude solveur | `4.595 s` | `1.031 s` | `2.600 s` | `0.806 s` | `1.669 s` |
| Mur-à-mur du cas | `615.39 s` | `260.87 s` | `553.09 s` | `299.55 s` | `351.97 s` |

IPOPT constitue ici la seule comparaison full/reduced appariée sur les 100
cycles : la full ajoute `0.0970 %` de fatigue et `0.0393 %` d'AUC. La
réduction accélère la médiane chaude de `4.46x` et le cas mur-à-mur de
`2.36x`. FATROP reduced retrouve la fatigue IPOPT reduced à `0.00086 %`,
ce qui fournit un second contrôle indépendant de l'optimum reduced.

| IPOPT, 100 RHO | Fatigue full | Fatigue reduced | Écart full/reduced | AUC full | AUC reduced | Capacité finale full/reduced |
|---|---:|---:|---:|---:|---:|---:|
| Biceps | `3793.7639` | `3789.5310` | `+0.1117 %` | `5.480885` | `5.477255` | `0.9003577 / 0.9003658` |
| Deltoïde antérieur | `204.6645` | `204.7708` | `-0.0519 %` | `1.428317` | `1.428655` | `0.9843771 / 0.9843514` |
| Deltoïde postérieur | `87.0216` | `87.0775` | `-0.0641 %` | `0.929778` | `0.930088` | `0.9919500 / 0.9919459` |
| Triceps | `262.2585` | `262.1180` | `+0.0536 %` | `1.466004` | `1.465333` | `0.9760446 / 0.9759632` |

Le Biceps explique `4.233` unités des `4.212` unités d'écart total; les
petites compensations des deltoïdes annulent une partie de ce surplus. La
capacité finale du Biceps ne diffère que de `8.15e-6`. La réduction ne
« gagne » donc pas artificiellement une grande réserve de fatigue après 100
cycles : elle sélectionne une solution très voisine avec une mécanique
beaucoup moins coûteuse à dériver et factoriser.

Le raffinement IPOPT initial de MadNLP déplace bien l'échec reduced du RHO 1
au RHO 99, mais ne supprime pas la barrière full au RHO 81. Les deux échecs
atteignent `maxiter=2000`, avec des résidus primaux respectifs de `6.34e-2`
et `1.78e-2`. IPOPT converge sur les mêmes RHO et établit leur faisabilité;
le RHO suivant converge aussi avec MadNLP, mais depuis l'état du solve échoué.
En full, cette propagation change notamment la vitesse initiale de
`-7.420 rad/s` à `-6.489 rad/s`. Cette dernière valeur reste dans les bornes;
c'est le primal complet et ses contraintes, violées jusqu'à `2.29e-1`, qui
ne sont pas admissibles. Les coutures postérieures sont exactement nulles
parce que ce terminal invalide est copié au RHO suivant, non parce que le
cycle manquant est devenu faisable. Ces fenêtres sont informatives sur le
comportement du solveur, mais ne constituent donc pas une chaîne physique de
remplacement.

Les patrons de stimulation renforcent l'hypothèse d'un changement d'ensemble
actif. Sur la trajectoire IPOPT full valide, la PW du Biceps saute d'environ
`360 us` entre les cycles 82 et 83, puis effectue un saut presque symétrique
au cycle suivant. En reduced, les variations intercycles du Triceps sont
fortes et intermittentes, puis atteignent environ `43 us` avant l'échec
MadNLP au RHO 99; IPOPT franchit la même région avec des variations d'environ
`40–49 us`. Cela est compatible avec des optima presque dégénérés ou des
basculements de recrutement, plutôt qu'avec un défaut de la dynamique full
seule. Ce n'est toutefois pas une preuve formelle de causalité tant que les
vecteurs PW complets, leurs masques actifs à `pd0`/`600 us` et les
multiplicateurs du RHO échoué ne sont pas exportés.

Les coûts et fatigues MadNLP ne doivent pas servir à comparer full et reduced
à 100 RHO, car leurs préfixes physiques n'ont pas la même longueur. La
prochaine expérience MadNLP pertinente est une seconde chance au **même** RHO
depuis une primale reconstruite ou raffinée, et non un raffinement unique payé
seulement avant le RHO 1. Le benchmark ne devra plus avancer depuis un solve
non convergé : les checkpoints full RHO 80 et reduced RHO 98 doivent rester
les deux états de départ certifiés pour toutes les tentatives suivantes.

Enfin, ACADOS full avec un poids de cadence `1` résout numériquement
`96/100` fenêtres dans ce run, mais son préfixe strict s'arrête au RHO 13.
Les échecs aux RHO 14, 16, 18 et 20 atteignent `MAXITER`; les RHO 21 à 100
réussissent. Sa médiane chaude vaut `0.395 s` et la violation maximale de
vitesse sécante n'est plus que `0.0118 rad/s`. La vitesse est donc
prometteuse, mais la robustesse de transfert n'est pas encore suffisante pour
une comparaison physiologique sur 100 cycles. Le test A/B de rejet du rollout
et le raffinement de warm-start sont suivis dans des runs séparés; leurs
résultats ne doivent pas être mélangés aux nombres certifiés ci-dessus.

Le premier run A/B
[`30597927290`](https://github.com/mickaelbegon/cocofest/actions/runs/30597927290)
n'a pas exécuté la garde : le workflow envoyait le nom interne
`--acados-transfer-rollout-max-bound-violation`, absent du CLI public, au lieu
de `--shared-transfer-rollout-max-bound-violation`. Les cinq autres cas ont
produit leurs JSON, puis le contrôle d'intégrité a correctement rejeté
l'artefact incomplet. Cet échec est purement infrastructurel et ne fournit
aucune donnée scientifique sur la garde; le nom d'option est corrigé avant la
relance.

La relance corrigée est incluse dans le run
[`30599827365`](https://github.com/mickaelbegon/cocofest/actions/runs/30599827365).
Elle montre que la garde absolue `0.2` ne résout pas le problème :

- le cas full cadence-régularisé sans garde et le cas gardé réussissent tous
  deux `96/100` solves, avec le même préfixe physique strict de `13`;
- les premiers échecs restent exactement les RHO `14`, `16`, `18` et `20`;
- leur fatigue exécutée sur le préfixe vaut respectivement `132.646886` et
  `132.646876`, donc les deux trajectoires sont numériquement confondues;
- les médianes chaudes valent `0.420 s` et `0.396 s`, différence trop faible
  pour conclure sur un seul runner.

Le journal confirme que la garde rejette tous les rollouts IRK observés. Avant
un échec, la violation terminale d'angle vaut environ `0.25–0.31 rad`; après
un `MAXITER`, le rollout suivant peut présenter près de `9.7 rad/s` de
violation de cadence. Le seuil mélange toutefois angle, vitesse et états
musculaires dans une même norme brute. Son échec A/B indique que le rollout
n'est pas la cause unique : le transfert par extrapolation vers la branche
active suivante reste lui aussi insuffisant. La prochaine variante doit donc
utiliser des violations normalisées par les bornes et un retry conditionnel
depuis le meilleur itéré du `MAXITER`, avec mémoire ACADOS réinitialisée, au
lieu d'abaisser encore ce seuil absolu.

La première exécution de ce retry,
[`30601544564`](https://github.com/mickaelbegon/cocofest/actions/runs/30601544564),
n'est pas un résultat scientifique : le CLI du solveur principal acceptait
les quatre nouvelles options, mais le wrapper
`cycling_fes_solver_comparison.py` ne les déclarait pas. Les cinq autres cas
ont produit leur JSON, puis le contrôle d'intégrité a rendu le job rouge parce
que le sixième JSON manquait. Le wrapper transmet désormais explicitement
`store_iterates`, le nombre de retries, leur budget SQP et le seuil de
faisabilité; deux tests couvrent à la fois le parsing et la propagation
jusqu'à la configuration ACADOS.

La relance effective,
[`30602407146`](https://github.com/mickaelbegon/cocofest/actions/runs/30602407146),
est verte mais réfute cette première stratégie :

| ACADOS full, cadence poids 1 | Sans retry | Retry du meilleur itéré |
|---|---:|---:|
| Solves réussis | `96/100` | `96/100` |
| Préfixe physique strict | `13` | `13` |
| Premiers échecs | `14, 16, 18, 20` | `14, 16, 18, 20` |
| Fatigue du préfixe | `132.646886` | `132.646886` |
| Médiane chaude solveur | `0.565 s` | `0.563 s` |
| Mur-à-mur | `144.42 s` | `147.13 s` |

Le retry se déclenche bien quatre fois. Les meilleures faisabilités stockées
valent respectivement `8.80e-4`, `2.02e-3`, `1.64e-3` et `1.86e-5`, mais les
20 SQP supplémentaires terminent toutes encore avec `MAXITER`. Une bonne
faisabilité intermédiaire n'est donc pas un warm-start suffisant : après
réinitialisation, les duaux, la stationnarité et l'information de branche ne
sont pas restaurés. Le diagnostic confirme ce mécanisme : la stationnarité du
meilleur itéré vaut seulement `0.205`, `0.243`, `0.254` et `0.074`, mais
remonte à environ `395–481` au départ du retry après le reset. Les 20 SQP
reconstruisent alors un système KKT incohérent et terminent avec une
stationnarité de `0.311–0.405`.

Augmenter aveuglément le seuil ou le budget serait donc injustifié. La
prochaine ablation minimale doit restaurer, après le reset QP, l'itéré ACADOS
complet `(x,u,pi,lam,sl,su)` au même indice, en gardant exactement le budget
de 20 et les tolérances actuelles. Le candidat devra ensuite être choisi par
un critère multi-résidu : faisabilité inférieure à `2.5e-3`, puis
stationnarité minimale. Seulement si ce retry primal-dual échoue, une courte
phase `FEASIBILITY_QP` ou Byrd–Omojokun avant le SQP nominal deviendra
pertinente. Aucun itéré presque faisable ne doit être promu : le critère de
succès reste statut zéro et audit physique complet.

Conclusion : la mécanique reduced est maintenant validée comme approximation
physiologique de la full sur 100 RHO pour ce régime, avec un écart inférieur
à `0.10 %` et un gain chaud de `4.46x` sous IPOPT. Le point
d'incertitude principal n'est plus la réduction mécanique, mais la robustesse
des transferts MadNLP et ACADOS sur les branches actives du problème non
convexe.

### 19.5 Sweep expérimental MadNLP/MUMPS MX en full horizon

Le workflow accepte le mode `cycles=full_horizon` pour estimer la taille
maximale d'un OCP monolithique sur une classe de machine donnée. Ce mode est
isolé du benchmark RHO principal. Le RHO IPOPT reduced qui construit la seed
reste en SX, configuration déjà certifiée et plus rapide; seuls les OCP
monolithiques MadNLP travaillent en MX, car leur question est la capacité
mémoire d'un grand graphe plutôt que le meilleur temps par RHO.

La procédure est :

1. résoudre la séquence reduced d'un cycle par RHO jusqu'au plafond demandé;
2. exporter le préfixe convergé et physiquement valide, même si un RHO
   ultérieur échoue, et adapter le plafond à sa longueur;
3. concaténer cette trajectoire en une seed multi-cycle;
4. tester une grille sparse
   `1, 2, 3, 5, 10, 15, 20, 25, 30`, puis `35…60` par pas de 5 et
   `70, 80, …`;
5. donner deux chances à chaque non-convergence; continuer la grille après un
   trou de convergence, mais raffiner cycle par cycle après la première limite
   mémoire;
6. interrompre le groupe de processus avant épuisement de la mémoire.

Chaque taille repart du préfixe reduced correspondant, relevé exactement vers
la mécanique full. Le résultat full précédent est sauvegardé comme artefact,
mais n'initialise pas la taille suivante. Il s'agit donc explicitement d'un
**homotopie de taille appariée par RHO**, et non d'une continuation de branche
full-to-full. Cette indépendance évite de confondre capacité mémoire et
qualité d'une continuation full, au prix de constructions froides répétées.

La limite automatique de RSS vaut `12.5 GiB` sur un runner de `16 GiB` et
`97.5 GiB` sur une machine de `128 GiB`. Le moniteur additionne le RSS de tout
le groupe de processus MadNLP/Julia et conserve les logs et checkpoints de
chaque taille. Un cas n'est déclaré réussi que si :

$$
\text{statut solveur valide}
\;\land\;
\text{faisabilité primale}
\;\land\;
\text{audit physique}
\;\land\;
N_\mathrm{cycles\ certifiés}=N_\mathrm{cycles\ demandés}.
$$

Le comptage single-shot a été corrigé en conséquence : un OCP de 30 cycles
réussi exporte et certifie 30 cycles, et non une seule fenêtre de solveur.
Pour chaque taille, le pont IPOPT/MX optionnel part du préfixe RHO reduced,
le relève sur la mécanique full, puis tente de raffiner ce relevé avant
MadNLP/MUMPS. Chaque tentative est limitée à 30 minutes et une erreur de CLI,
d'import, de JSON ou le warning
`linear_solver ... unknown type mumps` rend le job rouge; elle n'est jamais
interprétée comme une limite scientifique. Ce mode mesure donc le plus grand
horizon testé qui converge, avec les éventuels trous explicitement listés,
mais pas encore un gain de performance exploitable en temps réel.

Commande GitHub pour un runner standard :

```bash
gh workflow run cycling_solver_benchmark_linux.yml \
  --repo mickaelbegon/cocofest \
  --ref codex/acados-pr-refresh \
  -f cycles=full_horizon \
  -f full_horizon_max_cycles=100 \
  -f full_horizon_memory_limit_gib=auto
```

La première campagne complète,
[`30600222246`](https://github.com/mickaelbegon/cocofest/actions/runs/30600222246),
sépare nettement limite mémoire et limite numérique :

| Horizon full/MX | Résultat | Itérations MadNLP | Résidu primal final | Pic RSS | Mur-à-mur |
|---:|---|---:|---:|---:|---:|
| 1 cycle | certifié | `110` | `1.89e-12` | `1.705 GiB` | `111.6 s` |
| 2 cycles, chance 1 | échec solveur | `2000` | `1.668` | `2.349 GiB` | `1536.9 s` |
| 2 cycles, chance 2 | échec solveur | `2000` | `1.668` | `2.352 GiB` | `1586.9 s` |
| 3 cycles | erreur d'infrastructure avant solve | — | — | `0.180 GiB` | `57.5 s` |

Le RHO reduced/SX construit pourtant les 100 cycles en `196.6 s`, avec un pic
RSS de seulement `0.383 GiB`. Le full horizon 1 cycle est déjà beaucoup plus
lent que le RHO full/SX : son raffinement IPOPT coûte `7.8 s`, puis MadNLP
`39.9 s`, auxquels s'ajoutent la construction MX et la préparation. À deux
cycles, le raffinement IPOPT limité à 300 itérations échoue avec
`inf_pr=3.145` et n'est pas appliqué; MadNLP part donc du simple relevé
reduced→full, atteint 2000 itérations et reste très infaisable. Les deux
chances identiques reproduisent le même bassin et n'apportent aucune
robustesse.

La campagne ne donne donc **aucune extrapolation mémoire fiable** vers 16 ou
128 GiB : la barrière observée est la qualité du warm-start à deux cycles,
très loin de la limite RSS de `12.5 GiB`. Le cas 3 cycles est une erreur
indépendante : le warm-up historique contient 60 contrôles, alors que le code
en exigeait 90. Le runner full horizon ignore désormais ce warm-up à taille
fixe et donne au raffinement IPOPT/SX le même budget de 2000 itérations que
MadNLP. Le prochain gate est volontairement limité à trois cycles. S'il ne
certifie toujours pas deux cycles, il faudra construire une seed full par RHO
avant de concaténer, plutôt que d'augmenter encore les budgets. Une ablation
monolithique **reduced/MX** doit aussi être exécutée séparément; elle répondra
à la question de capacité mémoire du modèle réduit sans la confondre avec le
pont mécanique full.

Le premier gate corrigé,
[`30603331596`](https://github.com/mickaelbegon/cocofest/actions/runs/30603331596),
s'est arrêté avant le solveur en `20.0 s`. Il a révélé une seconde erreur de
contrat, et non une limite MadNLP : le préfixe RHO certifié déclarait
correctement `warmup_cycles_consumed=1`, alors que le consommateur full, qui
désactive désormais ce warm-up redondant, attendait `0`. Réécrire le seed à
`0` aurait effacé sa chronologie de fatigue. Le runner passe donc maintenant
un mode explicite qui adopte la valeur du seed avant la validation complète
des autres métadonnées. Les valeurs absentes, négatives, booléennes ou non
entières restent rejetées. Cette correction conserve à la fois l'absence de
construction redondante et la provenance physique du seed; elle est couverte
par les tests de CLI, de forwarding et de validation de métadonnées.
