# Prompt de continuation

Copier le bloc suivant dans une nouvelle tâche Codex après avoir cloné la
branche sur la machine Linux. Joindre les premiers logs ou le lien du run CI
si la campagne a déjà commencé.

```text
Reprends le benchmark des solveurs du RHO de pédalage FES sur la branche
codex/acados-pr-refresh de mickaelbegon/cocofest.

Commence impérativement par lire, dans cet ordre :
1. docs/cycling_solver_benchmark/README.md
2. docs/cycling_solver_benchmark/resume_and_todo.md
3. docs/cycling_solver_benchmark/linux_32core_setup.md
4. uniquement les sections pertinentes de
   docs/cycling_solver_benchmark/development_history.md

Inspecte ensuite git status, le HEAD Cocofest et les SHA effectifs de Bioptim,
ACADOS, libMad et CasADi. Préserve tous les changements utilisateur et ne
réutilise aucun artefact dont la provenance ou la transcription diffère.

Principe scientifique non négociable : l'ancienne référence Radau degré 3
n'est plus l'oracle. La force passive doit rester active. La dynamique du
calcium doit être raffinée et validée contre le régime périodique analytique.
Il faut accélérer le problème corrigé, pas reproduire un ancien coût obtenu
avec une erreur de modèle ou une discrétisation insuffisante.

Objectif de cette reprise :
- conserver les profils verrouillés scientific-radau4/5/6 et vérifier leur
  hash distinct;
- partir du gate Linux 30750686602 : IPOPT et MadNLP passent `5/5`, mais
  R5--R6 diffère encore de `0.343–0.398 %` sur la fatigue et de
  `0.580–0.645 %` sur l'AUC;
- ne pas retenir Radau 4 comme méthode finale : son erreur calcique isolée est
  `0.4518 %`, malgré son intérêt comme témoin rapide;
- transférer les contrôles R4/R5/R6 entre transcriptions et les PW full/reduced
  entre formulations;
- réintégrer exactement les mêmes PW avec un intégrateur dense commun avant
  toute nouvelle optimisation;
- comparer IPOPT/MUMPS et MadNLP/MUMPS reduced sur le même NLP raffiné;
- conserver les contrôles full Radau 5 IPOPT et MadNLP : MadNLP full/reduced
  s'accorde à `0.00194 %`, mais IPOPT full trouve une branche `0.400 %` plus
  basse que son reduced;
- ne jamais propager le terminal d'un RHO non certifié;
- garder le gate 30 bloqué jusqu'à ce que le transfert croisé et la
  réintégration dense expliquent l'écart R5--R6;
- mesurer construction, compilation, préparation, temps solveur et temps mural
  chaud séparément;
- reporter coût, fatigue exécutée, AUC et capacité finale pour les quatre
  muscles, ainsi que les PW aux cycles 10, 30 et 100;
- documenter toute divergence full/reduced supérieure à 0.1 % par muscle.

Sur la machine 32 cœurs :
- utiliser CMAKE_BUILD_PARALLEL_LEVEL=32 et BENCHMARK_THREADS=32;
- garder d'abord OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1,
  MKL_NUM_THREADS=1, NUMEXPR_NUM_THREADS=1 et JULIA_NUM_THREADS=1 pour la
  baseline reproductible;
- ne faire le sweep MUMPS 1/2/4/8/16/30/32 threads qu'après certification de
  la baseline;
- exécuter les solveurs successivement sur une machine unique pour éviter la
  contention, avec artefacts séparés.

Pour ACADOS, partir du résultat certifié aux nœuds du run `30763188906` : la
garde full `2.60` avec Phase-I mécanique atteint `100/100`, avec `0.739 s` de
médiane et `0.909 s` au P90 lorsque le coût de projection est inclus. Ne pas
remplacer ce temps par la seule médiane solveur (`0.105 s`). Byrd--Omojokun et
le shift simple restent limités à 80 RHO; la Phase-I sur tous les états est
rejetée parce qu'elle déplace les états Ding et produit un fort drift DOP853.
La prochaine piste est un audit apparié des mêmes PW : rollout full remis à
l'état certifié à chaque RHO, replay reduced, puis stabilisation/projection du
contact full si nécessaire. Ajouter ensuite un écran bon marché avant la
Phase-I mécanique, puisque 65 projections sur 99 sont rejetées après calcul.
RTI vient seulement après cette certification continue.

Ne rouvre pas PARDISO/MadNLP, Alpaqa, FATROP/RK4 ou un surrogate neuronal sans
un élément nouveau. MUMPS reste le backend MadNLP. FATROP full n'est plus un
échec attendu : Bioptim `4179bf07` corrige le rangement stage-wise et le smoke
local passe `1/1`; il faut maintenant le certifier en Linux avec les mêmes
audits que FATROP reduced.

Utilise un second agent pour auditer les choix numériques, les critères de
convergence du calcium et l'interprétation des résultats full/reduced. Fais les
modifications par petites étapes testées. Après chaque gate Linux, lis les
logs et les JSON complets avant de poursuivre. Mets à jour README.md,
resume_and_todo.md, development_history.md et le message pour Kevin avec les
résultats certifiés seulement.

À la fin : donne le tableau comparatif, les échecs et leur premier RHO, les
limites restantes et le prochain TODO précis. Commit et pousse uniquement les
changements testés sur codex/acados-pr-refresh.
```
