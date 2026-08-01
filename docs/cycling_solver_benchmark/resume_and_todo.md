# Point de reprise du benchmark RHO

État consolidé au 31 juillet 2026 sur la branche
`codex/acados-pr-refresh`. Le HEAD précédant cette mise à jour documentaire est
`2aa3633374074f5da7c2e6800a2bfd166bbe6b7b`.

Ce document répond à deux questions :

1. quelles conclusions peuvent déjà être considérées comme acquises;
2. quelles tâches doivent être reprises, dans quel ordre et avec quel critère
   de succès.

Pour les justifications complètes et les liens vers chaque run, consulter
l'[historique des développements](development_history.md). La méthode active
est décrite dans le [README](README.md).

Pour reprendre sur un autre calculateur, copier le
[prompt de continuation](continuation_prompt.md) et suivre la
[procédure Linux 32 cœurs](linux_32core_setup.md).

## 1. Configuration scientifique courante

- Objectif : minimiser uniquement la fatigue musculaire.
- Couple externe : `0 N.m`; aucune assistance ne masque l'arrivée de la
  fatigue.
- Un RHO résout un OCP d'un cycle, avec 30 décisions de PW par muscle.
- Quatre muscles, soit 20 états de Ding.
- PW admissibles dans $[pd0,600\,\mu\mathrm{s}]$.
- Angle terminal absolu, sans accumulation de la solution du cycle précédent.
- Graphes SX pour les campagnes actives.
- MUMPS pour IPOPT et MadNLP.
- Mécanique reduced : deux états $[\theta,\omega]$, sans imposer une cadence
  constante.
- Mécanique full : six états mécaniques, utilisée comme contrôle scientifique.
- Force passive active en full et en reduced.

Une chaîne est valide seulement jusqu'au premier RHO qui échoue au statut
solveur ou à l'audit physique. Les fenêtres qui convergent après cet échec
restent diagnostiques et ne sont pas agrégées comme endurance.

## 2. Conclusions considérées comme acquises

### 2.1 La référence historique n'est plus l'oracle

Deux défauts empêchent de prendre l'ancienne transcription comme cible
scientifique :

1. un ancien chemin `updating_model()` pouvait perdre l'activation de la
   relation de force passive;
2. la collocation Radau degré 3 sous-estime de `6.39 %` le calcium périodique
   isolé étudié.

La nouvelle règle est de corriger le modèle ou la transcription, puis de
recertifier les solveurs sur ce problème commun. Un ancien coût n'est conservé
que comme donnée historique.

### 2.2 La force passive est maintenant conservée

Le chemin full transmet explicitement
`activate_passive_force_relationship`. La mécanique reduced l'active par
défaut et évalue le même coefficient périodique. Un test ciblé vérifie que la
mise à jour du modèle conserve simultanément les relations force-longueur,
force-vitesse et passive.

Une campagne sans force passive doit désormais porter le statut d'ablation.
Elle ne doit pas être mélangée à une comparaison de solveurs.

### 2.3 La résolution du calcium doit être raffinée

Pour le régime périodique isolé :

| Méthode | $C_N$ |
|---|---:|
| Analytique | `0.162982158353` |
| ACADOS IRK, 4 étages et 5 sous-pas | `0.162982158637` |
| Radau degré 3 | `0.152573519058` |
| ERK testé | `0.232903256` |

Radau 3 est numériquement cohérent avec sa transcription, mais insuffisamment
fidèle à la dynamique calcique. ERK est exclu avec les réglages testés. La
bonne approche est de garder les 30 décisions de PW tout en augmentant la
résolution interne des états, par Radau 5 ou sous-pas IRK.

MadNLP/MUMPS Radau 5 a convergé `5/5`, avec une médiane chaude proche de
`1.99 s`. IPOPT Radau 5 a produit des points primalement faisables, mais un
RHO est resté non optimal après 2000 puis 5000 itérations. Le raffinement
n'est donc pas encore certifié à 30 ou 100 RHO.

### 2.4 La réduction mécanique est validée pour le régime testé

La réduction conserve exactement les 20 états de Ding et remplace seulement
la mécanique par :

```math
q=\Phi(\theta),
\qquad
\dot q=T(\theta)\omega.
```

La campagne IPOPT appariée à 100 RHO a donné :

| IPOPT/MUMPS | Full | Reduced |
|---|---:|---:|
| Préfixe physique | `100` | `100` |
| Fatigue exécutée | `4347.708565` | `4343.497299` |
| Écart relatif |  | environ `-0.097 %` |
| Médiane chaude | `4.595 s` | `1.031 s` |
| Gain reduced |  | `4.46x` |

La capacité finale du Biceps ne diffère que d'environ `8.15e-6`. Le grand
écart de fatigue observé dans les premières campagnes provenait donc de
différences de transcription et de bornes, pas d'une réserve artificielle
créée par la réduction.

Cette conclusion reste à confirmer sur la transcription calcique raffinée.

### 2.5 SX et compilation persistante sont les leviers logiciels sûrs

- SX réduit de `57.5 %` à `60.5 %` la médiane chaude par rapport à MX sur les
  comparaisons appariées.
- La bibliothèque C doit être construite une fois et réutilisée pendant tous
  les RHO.
- L'état initial, l'angle terminal et les bornes mobiles doivent être des
  paramètres runtime.
- Le temps de construction et de compilation doit être séparé du temps chaud
  du RHO.

Augmenter le nombre de cœurs ne multipliera pas la vitesse par le nombre de
cœurs. La construction de graphes, de nombreuses évaluations CasADi et une
partie de l'algorithme restent sérielles. Le parallélisme est surtout utile
entre solveurs ou formulations; le gain intra-solveur dépend de la part de la
factorisation MUMPS réellement parallélisable.

## 3. État des solveurs

### 3.1 IPOPT/MUMPS

- Seul solveur ayant certifié full et reduced sur les 100 RHO appariés.
- Reduced/SX compilé : environ `1.0 s` chaud par RHO sur Radau 3.
- Choix robuste actuel pour produire une chaîne complète.
- Radau 5 reste sensible à la stationnarité malgré une bonne faisabilité
  primale.

### 3.2 MadNLP/MUMPS

- MUMPS est transmis à libMad sous le nom typé `MumpsSolver`.
- PARDISO/MKL n'a pas apporté de gain et est abandonné.
- Sur 100 RHO Radau 3, le full a un préfixe strict de 80 et le reduced de 98.
- Les RHO suivants peuvent converger, mais ils partent alors d'un état non
  certifié et ne forment pas une endurance valide.
- Radau 5 est prometteur : `5/5` et environ `1.99 s` chaud.
- La compilation persistante fonctionne techniquement, mais son gain doit être
  remesuré sur une campagne assez longue pour amortir le coût initial.

Le prochain levier MadNLP n'est pas un budget d'itérations plus élevé. Il faut
reconstruire une primale admissible au **même RHO** après l'échec, puis repartir
du dernier checkpoint certifié.

### 3.3 FATROP

- Reduced/SX/collocation : `100/100` et solution physiologique cohérente avec
  IPOPT reduced.
- Plus lent qu'IPOPT reduced dans la campagne active.
- Full : échec structurel avant le solve lors de l'identification des gaps de
  collocation; ce n'est pas une preuve d'infaisabilité du problème.
- RK4 est abandonné.

FATROP reste un excellent contrôle indépendant de l'optimum reduced, mais pas
le meilleur chemin de production actuel.

### 3.4 ACADOS 0.5.5

ACADOS est le seul candidat ayant régulièrement résolu une fenêtre en environ
`0.4–0.6 s`. Il n'est toutefois pas robuste sur la chaîne : le meilleur
préfixe physique full reste limité à 13 RHO, avec des échecs aux RHO 14, 16,
18 et 20.

Les analyses suivantes ont été menées :

- pénalité douce de cadence;
- homotopie terminale;
- retry primal seul puis primal-dual complet;
- `SQP_WITH_FEASIBLE_QP` avec Byrd–Omojokun;
- préservation sélective des multiplicateurs;
- sélection entre shift projeté et rollout IRK projeté.

Aucune n'a déplacé le premier échec. Byrd–Omojokun réduit parfois les résidus
d'environ un ordre de grandeur, mais reste au-dessus du seuil physique. La
préservation des duals est instable. L'homotopie multiplie le temps effectif
sans prolonger le préfixe. Le rollout IRK concurrent ajoute environ `0.245 s`
par transfert et est rejeté dans la plupart des cas.

La conclusion est que le prochain levier doit améliorer le **primal transféré
sur les équations discrètes**, et non ajouter un nouveau mécanisme de
globalisation autour du même seed.

### 3.5 Alpaqa et autres voies archivées

- Alpaqa : intégration non fonctionnelle sur cette formulation.
- PARDISO pour MadNLP : aucun gain démontré.
- MA57 : résultat IPOPT historique, non retenu dans la matrice portable.
- Surrogate neuronal : non prioritaire avant profilage détaillé.
- Full horizon MadNLP/MX : question distincte du RHO temps réel.

## 4. Full horizon MadNLP/MX

Le monolithique full/MX certifie un cycle. Deux cycles échouent très loin des
limites mémoire : environ `2.35 GiB` de RSS sur une limite CI de `12.5 GiB`.
Le témoin reduced/MX échoue lui aussi à deux cycles. La cause principale n'est
donc pas la mémoire ni uniquement la mécanique full, mais la construction et
la couture du seed multi-cycle avec la transcription non convexe.

Ce chantier ne doit pas bloquer l'optimisation du RHO. Il ne redevient
prioritaire que si l'objectif est explicitement un horizon monolithique.

## 5. TODO priorisés

### P0 — Définir et certifier le problème scientifique corrigé

- [ ] Ajouter un profil nommé `scientific-radau5` ou `irk-refined`, distinct
  de `legacy-radau3`.
- [ ] Enregistrer explicitement dans chaque JSON le statut de la force passive,
  le degré de collocation, le nombre d'étages et les sous-pas.
- [ ] Ajouter un test isolé du calcium périodique contre la valeur analytique.
- [ ] Découpler les 30 décisions de PW de la résolution interne des états.
- [ ] Effectuer une étude de convergence Radau 3, Radau 5 et raffinement
  suivant, avec erreur calcium, fatigue, AUC et temps.
- [ ] Lancer `5`, puis `30`, puis `100` RHO seulement lorsque le gate précédent
  est physiquement certifié.

Critère de sortie proposé : erreur relative du calcium inférieure à `0.1 %`,
variation de fatigue et d'AUC inférieure à `0.1 %` au raffinement suivant, et
aucune violation interne des bornes.

### P1 — Établir la nouvelle baseline de performance

- [ ] Comparer IPOPT/MUMPS et MadNLP/MUMPS reduced sur exactement le même NLP
  raffiné.
- [ ] Conserver IPOPT full comme contrôle apparié aux cycles 5, 30 et 100.
- [ ] Réutiliser une seule bibliothèque compilée par cas et vérifier son hash.
- [ ] Séparer construction, compilation, préparation initiale, solveur chaud et
  temps mural effectif.
- [ ] Comparer fatigue et AUC des quatre muscles ainsi que les PW aux cycles
  10, 30 et 100.

Critère de sortie : un tableau unique où tous les solveurs ont le même modèle,
le même maillage des états, les mêmes PW admissibles et le même préfixe
physique.

### P1 — Rendre MadNLP robuste au premier RHO difficile

- [ ] Sauvegarder le dernier checkpoint physique avant l'échec full RHO 81 et
  reduced RHO 99.
- [ ] Ne jamais avancer la fenêtre depuis une solution non certifiée.
- [ ] Donner deux chances au même RHO depuis des primales distinctes et
  documentées : shift projeté, puis raffinement IPOPT ou restauration dédiée.
- [ ] Réutiliser la solution récupérée uniquement si statut et audits passent.
- [ ] Comparer les ensembles actifs PW, multiplicateurs, stationnarité et
  conditionnement avant/après récupération.

Critère de sortie : préfixe strict `100/100` sans propagation d'un terminal
invalide.

### P1 — Refaire le transfert ACADOS sur la dynamique discrète

- [ ] Construire une projection qui minimise directement les défauts de
  tir/collocation de la nouvelle fenêtre, avec priorité aux états mécaniques et
  au calcium.
- [ ] Conserver l'angle absolu et la variété de contact pendant cette
  projection.
- [ ] Comparer le candidat projeté au shift simple avant le solve, sans payer
  systématiquement un rollout complet.
- [ ] Si nécessaire, précompiler deux capsules synchronisées : restauration de
  faisabilité puis objectif de fatigue.
- [ ] N'évaluer RTI qu'après plusieurs chaînes SQP complètes certifiées.
- [ ] Réintégrer densément chaque intervalle accepté pour vérifier cadence et
  calcium entre les nœuds.

Critère de sortie intermédiaire : franchir le RHO 14 sans relâcher les
contraintes physiques. Critère final : `100/100` avec médiane effective sous
`1 s`.

### P2 — Recertifier full contre reduced sur le modèle corrigé

- [ ] Rejouer exactement les mêmes PW et les mêmes 20 états Ding dans les deux
  mécaniques.
- [ ] Comparer RHS musculaires, couple, accélération, force passive et calcium
  point par point.
- [ ] Refaire la comparaison IPOPT full/reduced à 5, 30 puis 100 RHO avec le
  calcium raffiné.
- [ ] Expliquer tout écart supérieur à `0.1 %` par muscle, pas seulement sur la
  somme.

Critère de sortie : coûts, AUC et capacités finales appariés, sans différence
de modèle cachée.

### P2 — Profiler avant de paralléliser davantage

- [ ] Mesurer séparément temps des fonctions, Jacobien, Hessienne,
  factorisation et backsolve.
- [ ] Tester MUMPS avec `1`, `2`, `4`, `8`, `16`, `30` et éventuellement `48`
  threads sur le même NLP chaud.
- [ ] Répéter chaque cas au moins trois fois et fixer les autres sources de
  parallélisme.
- [ ] Comparer le parallélisme intra-solveur au lancement parallèle des quatre
  familles de solveurs.

Critère de sortie : courbe de speedup et d'efficacité; ne pas extrapoler un
gain `30x` ou `48x` sans mesure.

### P3 — Nettoyer le langage et les artefacts du benchmark

- [ ] Remplacer dans les nouveaux cas le terme ambigu `reference` par
  `legacy-radau3`, `scientific-radau5` ou un nom descriptif équivalent.
- [ ] Garder les anciens noms uniquement dans l'archive et les liens de runs.
- [ ] Ajouter dans le rapport un badge `historique`, `certifié numérique` ou
  `certifié scientifique`.
- [ ] Faire échouer la CI si la force passive ou la transcription diffèrent
  entre deux cas annoncés comme appariés.
- [ ] Mettre à jour le message pour Kevin après la première campagne
  scientifique 30 RHO.

### P3 — Atteindre un échec réellement causé par la fatigue

- [ ] Après certification à 100 RHO, prolonger reduced vers 300 RHO.
- [ ] Si la chaîne reste faisable, prolonger vers 1000 RHO.
- [ ] Conserver deux chances de non-convergence au même RHO sans avancer depuis
  un échec.
- [ ] Distinguer clairement échec de fatigue, changement d'ensemble actif,
  limite d'itérations et erreur d'infrastructure.

## 6. Pistes à ne pas relancer sans nouvel élément

- PARDISO/MKL pour MadNLP;
- Alpaqa sur l'interface actuelle;
- FATROP/RK4;
- simple augmentation du budget SQP ACADOS;
- homotopie terminale ACADOS actuelle;
- retry ACADOS primal ou primal-dual inchangé;
- conservation globale des duals ACADOS;
- rollout IRK concurrent actuel;
- réseau neuronal avant d'avoir identifié une fonction dominante;
- estimation de la capacité full horizon à partir de la RAM avant d'avoir une
  seed multi-cycle certifiée.

Une piste archivée peut être réouverte uniquement si un changement précis de
modèle, d'interface ou d'algorithme invalide le résultat négatif précédent.

## 7. Première session de reprise conseillée

1. Vérifier le statut Git et les SHA effectifs de Cocofest, Bioptim et ACADOS.
2. Lire le [README actif](README.md), puis uniquement les sections historiques
   liées au calcium et à la force passive.
3. Implémenter le profil scientifique commun et son test analytique du calcium.
4. Lancer localement un RHO reduced IPOPT et MadNLP.
5. Déclencher le gate Linux 5 RHO.
6. N'ouvrir le gate 30 qu'après inspection des artefacts de convergence
   temporelle et de faisabilité.

La première question à trancher à la reprise est donc : Radau 5 suffit-il au
critère de convergence du calcium, ou faut-il raffiner encore les états sans
augmenter le nombre de décisions de PW?
